import random
from .rl_agents import DQNAgent, PPOAgent

class UAV:
    def __init__(self, uav_id, flight_height, device, is_leader=False, num_member_uavs=0, dqn_lr=5e-5, ppo_lr=5e-5, gamma=0.9):
        self.uav_id = uav_id
        self.is_leader = is_leader
        self.x = None
        self.y = None
        self.z = flight_height
        self.device = device
        self.agent = None
        self.offload_decision_agent = None

        if not self.is_leader:
            state_dim = 3
            action_dim = 2
            self.agent = DQNAgent(state_dim, action_dim, device=self.device, lr=dqn_lr, gamma=gamma)
            self.computing_capability_ghz = None
            self.cycles_per_bit = None
        else:
            state_dim_ppo = 5
            action_dim_ppo = num_member_uavs
            self.agent = PPOAgent(state_dim_ppo, action_dim_ppo, device=self.device, lr=ppo_lr, gamma=gamma)
            state_dim_dqn = 5
            action_dim_dqn = 2
            self.offload_decision_agent = DQNAgent(state_dim_dqn, action_dim_dqn, device=self.device, lr=dqn_lr, gamma=gamma)
            self.computing_capability_ghz = None
            self.cycles_per_bit = None

    def update_position(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        role = "Leader UAV" if self.is_leader else f"Member UAV {self.uav_id}"
        base_info = f"{role} | Pos: ({self.x}, {self.y}, {self.z})"
        if not self.is_leader:
            compute_info = (f" | Computing: {self.computing_capability_ghz:.2f} GHz | "
                            f"Cycles per bit(C_i): {self.cycles_per_bit} cycles/bit")
            return base_info + compute_info
        return base_info

class EdgeLayer:
    def __init__(self, num_member_uavs, area_width, area_height, flight_height, device, dqn_lr=5e-5, ppo_lr=5e-5, gamma=0.9):
        self.num_member_uavs = num_member_uavs
        self.area_width = area_width
        self.area_height = area_height
        self.flight_height = flight_height
        self.device = device
        self.leader_uav = UAV(uav_id='L', flight_height=self.flight_height, device=self.device, is_leader=True,
                              num_member_uavs=self.num_member_uavs, dqn_lr=dqn_lr, ppo_lr=ppo_lr, gamma=gamma)
        leader_x = self.area_width / 2
        leader_y = self.area_height / 2
        self.leader_uav.update_position(leader_x, leader_y)
        self.member_uavs = []
        self._create_all_member_uavs(dqn_lr, gamma)

    def _create_all_member_uavs(self, dqn_lr, gamma):
        for i in range(self.num_member_uavs):
            uav = UAV(uav_id=i + 1, flight_height=self.flight_height, device=self.device, dqn_lr=dqn_lr,
                      gamma=gamma)

            uav.computing_capability_ghz = 10.0
            uav.cycles_per_bit = random.randint(500, 1500)

            self.member_uavs.append(uav)

    def update_all_uavs_for_new_slot(self):
        occupied_positions = set()
        for uav in self.member_uavs:
            while True:
                new_x = random.randint(0, self.area_width)
                new_y = random.randint(0, self.area_height)
                if (new_x, new_y) not in occupied_positions:
                    uav.update_position(new_x, new_y)
                    occupied_positions.add((new_x, new_y))
                    break

    def get_total_member_computing_capability(self):
        return sum(uav.computing_capability_ghz for uav in self.member_uavs)

    def display_uavs_status(self):
        print(f"\n--- Edge Computing Layer Status (1 leader, {self.num_member_uavs} members) ---")
        print(self.leader_uav)
        for uav in self.member_uavs:
            print(uav)
