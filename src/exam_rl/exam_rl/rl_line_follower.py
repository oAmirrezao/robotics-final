import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path, Odometry
import numpy as np
import math
import torch
import os
from .ddpg import DDPG 

class RLLineFollower(Node):
    def __init__(self):
        super().__init__('rl_line_follower')
        
        # واسط‌های ROS
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Path, '/plan', self.path_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        self.path = []
        self.robot_pose = None
        self.robot_yaw = 0.0
        
        # تعریف فضای RL برای DDPG
        # State: [Cross_Track_Error, Heading_Error, Linear_Vel, Angular_Vel]
        self.observation_shape = 4 
        self.action_space = type('obj', (object,), {'n': 2}) # [v, w]
        self.observation_space = type('obj', (object,), {'n': 4})
        self.n_active_features = 1 # ضریب اسکیلینگ
        
        self.get_logger().info('RL Node Init. Waiting for Path...')

    def path_callback(self, msg):
        self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.get_logger().info(f'Path received with {len(self.path)} points.')

    def odom_callback(self, msg):
        self.robot_pose = msg.pose.pose.position
        # محاسبه Yaw از کواتریون
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def get_state(self):
        if not self.path or self.robot_pose is None:
            return np.zeros(self.observation_shape)
            
        # پیدا کردن نزدیک‌ترین نقطه مسیر به ربات
        min_dist = float('inf')
        closest_idx = 0
        rx, ry = self.robot_pose.x, self.robot_pose.y
        
        for i, (px, py) in enumerate(self.path):
            dist = math.sqrt((px - rx)**2 + (py - ry)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        # 1. خطای فاصله (Cross Track Error)
        cte = min_dist
        
        # 2. خطای زاویه (Heading Error)
        # زاویه مطلوب، بردار به سمت نقطه بعدی مسیر است
        next_idx = min(closest_idx + 3, len(self.path)-1) # نگاه به 3 نقطه جلوتر برای نرمی
        tx, ty = self.path[next_idx]
        desired_yaw = math.atan2(ty - ry, tx - rx)
        heading_error = desired_yaw - self.robot_yaw
        
        # نرمال کردن زاویه بین -pi و pi
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        
        # State برمی‌گردانیم
        return np.array([cte, heading_error, 0.0, 0.0])

    def step_env(self, action):
        # اجرای اکشن روی ربات
        twist = Twist()
        # کلیپ کردن مقادیر برای ایمنی
        twist.linear.x = float(np.clip(action[0], 0.0, 0.3)) 
        twist.angular.z = float(np.clip(action[1], -0.5, 0.5))
        self.cmd_vel_pub.publish(twist)
        
        # محاسبه پاداش و وضعیت بعدی (اینجا شبیه‌سازی گسسته زمان است)
        next_state = self.get_state()
        
        cte = next_state[0]
        he = next_state[1]
        
        # Reward Function: هرچه خطا کمتر، پاداش بیشتر
        reward = 1.0 - (5.0 * abs(cte) + 2.0 * abs(he))
        
        done = False
        # شرط پایان اپیزود (تصادف یا انحراف زیاد)
        if abs(cte) > 0.5: 
            reward = -10.0
            done = True
            
        # شرط رسیدن به هدف
        if self.path and self.is_at_end():
            reward = 100.0
            done = True
            
        return next_state, reward, done, False, {}

    def is_at_end(self):
        if not self.path: return False
        lx, ly = self.path[-1]
        dist = math.sqrt((lx - self.robot_pose.x)**2 + (ly - self.robot_pose.y)**2)
        return dist < 0.3

def main(args=None):
    rclpy.init(args=args)
    env_node = RLLineFollower()
    
    # ساخت ایجنت DDPG
    # نکته: ما env_node را پاس می‌دهیم چون DDPG شما پارامترها را از env می‌خواند
    agent = DDPG(env_node, buffer_size=10000)
    
    num_episodes = 20 # برای امتحان تعداد کم
    
    for ep in range(num_episodes):
        # ریست کردن محیط (در واقعیت باید ربات را برگردانیم، اینجا فقط متغیرها)
        state = env_node.get_state()
        ep_reward = 0
        steps = 0
        
        print(f"--- Start Episode {ep+1} ---")
        
        while steps < 500: # Max steps per episode
            # چرخش ROS برای دریافت داده‌های سنسور
            rclpy.spin_once(env_node, timeout_sec=0.05)
            
            # اگر مسیری نیست، صبر کن
            if not env_node.path:
                continue

            # انتخاب اکشن
            if ep < 2: # وارم آپ
                 action = np.random.uniform(-0.5, 0.5, 2)
                 action[0] = np.abs(action[0]) # سرعت خطی مثبت
            else:
                action_tensor = agent.select_action(state)
                action = action_tensor.cpu().numpy().flatten()
            
            # اعمال اکشن
            next_state, reward, done, _, _ = env_node.step_env(action)
            
            # ذخیره در بافر و آموزش
            if not np.all(state == 0): # ذخیره نکن اگر استیت خالیه
                agent.buffer.push(state, action, reward, next_state, done)
                agent.optimize()
            
            state = next_state
            ep_reward += reward
            steps += 1
            
            if done:
                # توقف ربات
                env_node.cmd_vel_pub.publish(Twist())
                break
        
        print(f"Episode {ep+1} finished. Total Reward: {ep_reward:.2f}")

    # ذخیره مدل نهایی
    torch.save(agent.actor.state_dict(), 'final_actor.pth')
    env_node.destroy_node()
    rclpy.shutdown()
