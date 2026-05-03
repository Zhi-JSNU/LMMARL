import random
from .rl_agents import PPOAgent

class TaskVehicle:
    def __init__(self, vehicle_id, device, lr=5e-5, gamma=0.9):
        self.vehicle_id = vehicle_id
        self.x = None
        self.y = None
        self.z = 0
        self.computing_capability_ghz = 2.0
        self.cycles_per_bit = random.randint(500, 1500)
        self.single_task_data_size_mb = 0
        self.tasks_per_slot = 0
        state_dim = 3
        action_dim = 1
        self.agent = PPOAgent(state_dim, action_dim, device=device, lr=lr, gamma=gamma)

    def update_position(self, x, y):
        self.x = x
        self.y = y

    def generate_tasks_for_new_slot(self, task_size_range=(1.0, 3.0), tasks_per_slot_range=(5, 10)):
        self.single_task_data_size_mb = random.uniform(*task_size_range)
        self.tasks_per_slot = random.randint(*tasks_per_slot_range)

    def get_total_task_size_in_slot(self):
        return self.single_task_data_size_mb * self.tasks_per_slot

    def __str__(self):
        return (f"Vehicle {self.vehicle_id} | "
                f"Pos: ({self.x}, {self.y}) | "
                f"Computing: {self.computing_capability_ghz:.2f} GHz | "
                f"Cycles per bit(C_k): {self.cycles_per_bit} cycles/bit | "
                f"Task size: {self.single_task_data_size_mb:.2f} MB | "
                f"Total task size: {self.get_total_task_size_in_slot():.2f} MB")


class ClientLayer:
    def __init__(self, num_vehicles, area_width, area_height, device, lr=5e-5, gamma=0.9):
        self.num_vehicles = num_vehicles
        self.area_width = area_width
        self.area_height = area_height
        self.device = device
        self.vehicles = []
        self._create_all_vehicles(lr, gamma)

    def _create_all_vehicles(self, lr, gamma):
        for i in range(self.num_vehicles):
            vehicle = TaskVehicle(vehicle_id=i + 1, device=self.device, lr=lr, gamma=gamma)
            self.vehicles.append(vehicle)

    def update_all_vehicles_for_new_slot(self, task_size_range=(1.0, 3.0), tasks_per_slot_range=(5, 10)):
        occupied_positions = set()
        for vehicle in self.vehicles:
            while True:
                new_x = random.randint(0, self.area_width)
                new_y = random.randint(0, self.area_height)
                if (new_x, new_y) not in occupied_positions:
                    vehicle.update_position(new_x, new_y)
                    occupied_positions.add((new_x, new_y))
                    break
            vehicle.generate_tasks_for_new_slot(task_size_range, tasks_per_slot_range)

    def display_vehicles_status(self):
        print(f"\n--- Client Layer Status ({self.num_vehicles} vehicles) ---")
        for vehicle in self.vehicles:
            print(vehicle)
