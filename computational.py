from typing import List, Dict
from .client_layer import TaskVehicle
from .edge_layer import UAV
from .fog_layer import GroundBaseStation


class ComputationModel:
    @staticmethod
    def calculate_client_layer_delay(vehicles: List[TaskVehicle], local_ratios: Dict[int, float]) -> (
    float, Dict[int, float]):
        max_delay = 0.0
        individual_delays = {}

        for vehicle in vehicles:
            alpha_k = local_ratios.get(vehicle.vehicle_id, 0.0)

            total_task_size_mb = vehicle.get_total_task_size_in_slot()
            total_task_size_bits = total_task_size_mb * 8 * 1024 * 1024
            cycles_per_bit = vehicle.cycles_per_bit
            total_required_cycles = total_task_size_bits * cycles_per_bit * alpha_k
            capability_ghz = vehicle.computing_capability_ghz
            capability_hz = capability_ghz * 1e9

            if capability_hz == 0:
                delay_k = float('inf')
            else:
                delay_k = total_required_cycles / capability_hz

            individual_delays[vehicle.vehicle_id] = delay_k

            if delay_k > max_delay:
                max_delay = delay_k

        return max_delay, individual_delays

    @staticmethod
    def calculate_edge_layer_delay(
            member_uavs: List[UAV],
            leader_to_member_results: Dict[int, Dict[str, float]]
    ) -> (float, Dict[int, Dict[str, float]]):
        max_total_delay = 0.0
        processing_details = {}

        for member in member_uavs:
            member_id = member.uav_id
            if member_id not in leader_to_member_results:
                continue

            tau_L_i = leader_to_member_results[member_id]['delay_sec']

            allocated_mb = leader_to_member_results[member_id].get('allocated_mb', 0)
            allocated_bits = allocated_mb * 8 * 1024 * 1024

            cycles_per_bit = member.cycles_per_bit
            capability_hz = member.computing_capability_ghz * 1e9

            if capability_hz == 0:
                computation_delay = float('inf')
            else:
                total_required_cycles = allocated_bits * cycles_per_bit
                computation_delay = total_required_cycles / capability_hz

            total_member_time = tau_L_i + computation_delay

            processing_details[member_id] = {
                "transmission_delay": tau_L_i,
                "computation_delay": computation_delay,
                "total_time": total_member_time
            }

            if total_member_time > max_total_delay:
                max_total_delay = total_member_time

        return max_total_delay, processing_details

    @staticmethod
    def calculate_fog_layer_computation_delay(
            base_station: GroundBaseStation,
            total_offloaded_data_mb: float
    ) -> float:
        data_bits = total_offloaded_data_mb * 8 * 1024 * 1024

        cycles_per_bit = base_station.cycles_per_bit
        capability_hz = base_station.computing_capability_ghz * 1e9

        total_required_cycles = data_bits * cycles_per_bit

        if capability_hz == 0:
            return float('inf')
        else:
            computation_delay = total_required_cycles / capability_hz
            return computation_delay
