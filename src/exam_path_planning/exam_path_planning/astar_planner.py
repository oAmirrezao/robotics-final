import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.srv import GetPlan
import numpy as np
import heapq
import math

class AStarPlanner(Node):
    def __init__(self):
        super().__init__('astar_planner')
        
        # اشتراک‌ها
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10)
        
        # پابلیشر مسیر
        self.path_pub = self.create_publisher(Path, '/plan', 10)
        
        # سرویس سرور (طبق خواسته سوال)
        self.srv = self.create_service(GetPlan, 'plan_path', self.plan_callback)

        self.map_data = None
        self.map_info = None
        self.current_pose = None
        
        self.get_logger().info('A* Planner Service is Ready on /plan_path')

    def map_callback(self, msg):
        # تبدیل داده فلت به ماتریس دو بعدی
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info

    def pose_callback(self, msg):
        self.current_pose = msg.pose.pose

    def world_to_grid(self, wx, wy):
        if self.map_info is None: return None
        gx = int((wx - self.map_info.origin.position.x) / self.map_info.resolution)
        gy = int((wy - self.map_info.origin.position.y) / self.map_info.resolution)
        return (gx, gy)

    def grid_to_world(self, gx, gy):
        if self.map_info is None: return None
        wx = (gx * self.map_info.resolution) + self.map_info.origin.position.x
        wy = (gy * self.map_info.resolution) + self.map_info.origin.position.y
        return (wx, wy)

    def heuristic(self, a, b):
        # فاصله اقلیدسی
        return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

    def get_neighbors(self, node):
        neighbors = []
        # حرکت در ۸ جهت (یا ۴ جهت طبق سلیقه استاد، اینجا ۴ جهت استاندارد است)
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        
        for dx, dy in directions:
            x, y = node[0] + dx, node[1] + dy
            
            if 0 <= x < self.map_info.width and 0 <= y < self.map_info.height:
                # بررسی برخورد: مقادیر > 65 اشغال هستند، -1 ناشناخته
                pixel_val = self.map_data[y, x]
                if 0 <= pixel_val < 65: # مسیر آزاد
                    neighbors.append((x, y))
        return neighbors

    def astar(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1] # برگرداندن مسیر از شروع به پایان

            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None

    def plan_callback(self, request, response):
        if self.map_data is None:
            self.get_logger().error('نقشه هنوز دریافت نشده است!')
            return response
        if self.current_pose is None:
            self.get_logger().error('موقعیت ربات (AMCL) نامشخص است!')
            return response

        # تبدیل مختصات
        start_grid = self.world_to_grid(self.current_pose.position.x, self.current_pose.position.y)
        goal_grid = self.world_to_grid(request.goal.pose.position.x, request.goal.pose.position.y)
        
        self.get_logger().info(f'Planning request: Start{start_grid} -> Goal{goal_grid}')

        # اجرای الگوریتم
        path_grid = self.astar(start_grid, goal_grid)

        if path_grid:
            path_msg = Path()
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = 'map'
            
            for p in path_grid:
                pose = PoseStamped()
                pose.header.frame_id = 'map'
                wx, wy = self.grid_to_world(p[0], p[1])
                pose.pose.position.x = wx
                pose.pose.position.y = wy
                pose.pose.orientation.w = 1.0 # جهت مهم نیست برای رسم مسیر
                path_msg.poses.append(pose)
            
            self.path_pub.publish(path_msg)
            response.plan = path_msg
            self.get_logger().info(f'مسیر با {len(path_grid)} نقطه پیدا و پابلیش شد.')
        else:
            self.get_logger().warn('مسیری پیدا نشد! (شاید مقصد داخل دیوار است)')
            
        return response

def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
