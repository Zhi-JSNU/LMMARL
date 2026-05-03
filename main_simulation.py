import random
import numpy as np
import torch
from .client_layer import ClientLayer
from .edge_layer import EdgeLayer
from .fog_layer import FogLayer
from .communication import CommunicationModel
from .computational import ComputationModel


def get_member_uav_state(member_uav, link_status_rate, total_offloaded_mb):
    return np.array([link_status_rate, member_uav.computing_capability_ghz, total_offloaded_mb], dtype=np.float32)


def get_leader_uav_state(total_offloaded_mb, active_members, link_statuses_cache):
    num_active = len(active_members)
    if num_active == 0:
        avg_f, avg_rate, task_per_ghz = 0, 0, 0
    else:
        avg_f = sum(m.computing_capability_ghz for m in active_members) / num_active
        avg_rate = sum(link_statuses_cache[m.uav_id]['rate_mbps'] for m in active_members) / num_active
        total_f = sum(m.computing_capability_ghz for m in active_members)
        task_per_ghz = total_offloaded_mb / total_f if total_f > 0 else 0
    return np.array([total_offloaded_mb, num_active, avg_f, avg_rate, task_per_ghz], dtype=np.float32)


def apply_ppo_action_and_get_results(active_members, total_offloaded_mb, allocation_ratios_dict, link_statuses_cache):
    if not active_members: return {}, 0.0
    active_ids = {m.uav_id for m in active_members}
    ratios = {mid: ratio for mid, ratio in allocation_ratios_dict.items() if mid in active_ids}
    total_ratio = sum(ratios.values())
    if total_ratio <= 0: return {}, 0.0
    norm_ratios = {mid: r / total_ratio for mid, r in ratios.items()}
    results = {}
    for m in active_members:
        ratio = norm_ratios.get(m.uav_id, 0.0)
        data = total_offloaded_mb * ratio
        rate_mbps = link_statuses_cache[m.uav_id]['rate_mbps']
        delay = CommunicationModel.calculate_transmission_delay(data * 8 * 1024 * 1024, rate_mbps * 1_000_000)
        results[m.uav_id] = {"allocated_mb": data, "delay_sec": delay}
    max_delay, _ = ComputationModel.calculate_edge_layer_delay(active_members, results)
    return results, max_delay


def run_single_simulation(
        num_vehicles, num_uavs, w1, w2, episodes=500, steps_per_episode=200,
        train_every_n_steps=4, ppo_epochs=10, debug=False, random_seed=42,
        task_size_range=(1.0, 3.0), tasks_per_slot_range=(5, 10),
        mode='proposed', dqn_lr=5e-5, ppo_lr=5e-5, gamma=0.9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if debug: print(f"--- Using device: {device}, seed: {random_seed} ---")
    AREA_WIDTH, AREA_HEIGHT, UAV_FLIGHT_HEIGHT = 500, 500, 100
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    station_positions = [(AREA_WIDTH * 0.25, AREA_HEIGHT * 0.25), (AREA_WIDTH * 0.75, AREA_HEIGHT * 0.25),
                         (AREA_WIDTH * 0.5, AREA_HEIGHT * 0.75)]
    client_layer = ClientLayer(num_vehicles=num_vehicles, area_width=AREA_WIDTH, area_height=AREA_HEIGHT, device=device,
                               lr=ppo_lr, gamma=gamma)
    edge_layer = EdgeLayer(num_member_uavs=num_uavs, area_width=AREA_WIDTH, area_height=AREA_HEIGHT,
                           flight_height=UAV_FLIGHT_HEIGHT, device=device, dqn_lr=dqn_lr, ppo_lr=ppo_lr, gamma=gamma)
    fog_layer = FogLayer(station_positions=station_positions)
    congestion_signal = 0.0
    TARGET_UPDATE_FREQUENCY = 10
    stable_phase_completion_status = []
    stable_phase_delays = []
    stable_phase_compute_delays = []
    stable_phase_start_episode = int(episodes * 0.9)
    rewards_history = []

    for episode in range(episodes):
        episode_reward = 0
        for step in range(steps_per_episode):
            client_layer.update_all_vehicles_for_new_slot(task_size_range, tasks_per_slot_range)
            edge_layer.update_all_uavs_for_new_slot()
            offload_ratios_dict, local_ratios_dict, vehicle_experiences = {}, {}, {}
            all_vehicle_states = np.array([
                [v.get_total_task_size_in_slot(), v.computing_capability_ghz, congestion_signal]
                for v in client_layer.vehicles
            ], dtype=np.float32)
            all_actions, all_log_probs, all_values = client_layer.vehicles[0].agent.select_action(all_vehicle_states)

            for i, v in enumerate(client_layer.vehicles):
                state = all_vehicle_states[i]
                action = all_actions[i]
                log_prob = all_log_probs[i]
                value = all_values[i]
                ratio = 1.0 / (1.0 + np.exp(-action))
                if mode == 'binary_offloading':
                    task_threshold = np.mean(task_size_range) * np.mean(tasks_per_slot_range)
                    ratio = 1.0 if v.get_total_task_size_in_slot() > task_threshold else 0.0
                offload_ratios_dict[v.vehicle_id] = float(ratio)
                local_ratios_dict[v.vehicle_id] = 1.0 - float(ratio)
                vehicle_experiences[v.vehicle_id] = (state, action, log_prob, value)
            total_offloaded, _, max_tx_delay = CommunicationModel.calculate_vehicle_to_leader_delays(
                client_layer.vehicles, edge_layer.leader_uav, offload_ratios_dict)
            link_statuses_cache = {
                m.uav_id: CommunicationModel.get_link_status_leader_to_member(edge_layer.leader_uav, m) for m in
                edge_layer.member_uavs}
            active_members, member_s_a = [], {}
            with torch.no_grad():
                for m in edge_layer.member_uavs:
                    rate_mbps = link_statuses_cache[m.uav_id]['rate_mbps']
                    state = get_member_uav_state(m, rate_mbps, total_offloaded)
                    action = m.agent.select_action(state).item()
                    member_s_a[m.uav_id] = {'state': state, 'action': action}
                    if action == 1: active_members.append(m)
            leader_state = get_leader_uav_state(total_offloaded, active_members, link_statuses_cache)
            decision = edge_layer.leader_uav.offload_decision_agent.select_action(leader_state).item()
            if mode == 'no_fog': decision = 0
            client_delay, _ = ComputationModel.calculate_client_layer_delay(client_layer.vehicles, local_ratios_dict)
            if decision == 0:
                if active_members:
                    if mode == 'no_collaboration':
                        best = max(active_members, key=lambda m: link_statuses_cache[m.uav_id]['rate_mbps'])
                        alloc_ratios = {best.uav_id: 1.0}
                        active_members = [best]
                        ppo_action, ppo_log_prob, ppo_value = None, None, None
                    else:
                        ppo_action, ppo_log_prob, ppo_value = edge_layer.leader_uav.agent.select_action(leader_state)
                        probs = torch.softmax(torch.from_numpy(ppo_action), dim=-1).numpy()
                        alloc_ratios = {m.uav_id: r for m, r in zip(edge_layer.member_uavs, probs)}
                    _, edge_proc_delay = apply_ppo_action_and_get_results(active_members, total_offloaded, alloc_ratios,
                                                                          link_statuses_cache)
                    offload_delay = max_tx_delay + edge_proc_delay
                else:
                    offload_delay = float('inf')
            else:
                nearest_bs = min(fog_layer.base_stations,
                                 key=lambda bs: CommunicationModel.calculate_distance(edge_layer.leader_uav, bs))
                rate_to_fog = \
                CommunicationModel.get_link_status_leader_to_base_station(edge_layer.leader_uav, nearest_bs)[
                    'rate_mbps']
                delay_to_fog = CommunicationModel.calculate_transmission_delay(total_offloaded * 8 * 1024 * 1024,
                                                                               rate_to_fog * 1_000_000)
                comp_delay_fog = ComputationModel.calculate_fog_layer_computation_delay(nearest_bs, total_offloaded)
                offload_delay = max_tx_delay + delay_to_fog + comp_delay_fog
            alpha = w1 / (w1 + w2 + 1e-9)
            total_system_delay = alpha * client_delay + (1.0 - alpha) * offload_delay
            DELAY_UPPER_BOUND = 100
            is_completed_step = True
            if (total_system_delay == float('inf') or np.isinf(
                    total_system_delay) or total_system_delay > DELAY_UPPER_BOUND):
                reward = -DELAY_UPPER_BOUND
                congestion_signal = DELAY_UPPER_BOUND
                is_completed_step = False
            else:
                reward = -total_system_delay
                congestion_signal = total_system_delay

            episode_reward += reward

            if episode >= stable_phase_start_episode:
                stable_phase_completion_status.append(1 if is_completed_step else 0)
                stable_phase_delays.append(total_system_delay if is_completed_step else DELAY_UPPER_BOUND)
                if is_completed_step:
                    compute_delay_step = total_system_delay - max_tx_delay
                    stable_phase_compute_delays.append(compute_delay_step)
                else:
                    stable_phase_compute_delays.append(DELAY_UPPER_BOUND)

            done = (step == steps_per_episode - 1)

            for v in client_layer.vehicles:
                state, action, log_prob, value = vehicle_experiences[v.vehicle_id]
                v.agent.store_experience(state, action, reward, log_prob, value, done)
            for m in edge_layer.member_uavs:
                info = member_s_a[m.uav_id]
                rate_mbps = link_statuses_cache[m.uav_id]['rate_mbps']
                next_state = get_member_uav_state(m, rate_mbps, total_offloaded)
                m.agent.store_experience(info['state'], info['action'], reward, next_state, done)
            next_leader_state = get_leader_uav_state(total_offloaded, active_members, link_statuses_cache)
            edge_layer.leader_uav.offload_decision_agent.store_experience(leader_state, decision, reward,
                                                                          next_leader_state, done)
            if mode != 'no_collaboration' and decision == 0 and active_members:
                edge_layer.leader_uav.agent.store_experience(leader_state, ppo_action, reward, ppo_log_prob, ppo_value,
                                                             done)
            if step > 0 and step % train_every_n_steps == 0:
                for m in edge_layer.member_uavs: m.agent.learn()
                edge_layer.leader_uav.offload_decision_agent.learn()
            if step > 0 and step % TARGET_UPDATE_FREQUENCY == 0:
                for m in edge_layer.member_uavs: m.agent.update_target_net()
                edge_layer.leader_uav.offload_decision_agent.update_target_net()

        avg_ep_reward = episode_reward / steps_per_episode
        rewards_history.append(avg_ep_reward)

        if client_layer.vehicles[0].agent.memory:
            client_layer.vehicles[0].agent.learn(epochs=ppo_epochs)
            for v in client_layer.vehicles[1:]:
                if hasattr(v.agent, 'clear_memory'):
                    v.agent.clear_memory()
                else:
                    v.agent.memory = []

        if mode != 'no_collaboration' and edge_layer.leader_uav.agent.memory:
            edge_layer.leader_uav.agent.learn(epochs=ppo_epochs)

        if debug and ((episode + 1) % 20 == 0 or episode == episodes - 1):
            print(f"  Episode {episode+1}/{episodes} | Avg Reward: {avg_ep_reward:.4f}")

    successful_delays = [d for d, c in zip(stable_phase_delays, stable_phase_completion_status) if c == 1]
    avg_delay_of_successful_tasks = np.mean(successful_delays) if successful_delays else float('inf')

    successful_compute_delays = [d for d, c in zip(stable_phase_compute_delays, stable_phase_completion_status) if
                                 c == 1]
    avg_compute_delay_of_successful_tasks = np.mean(successful_compute_delays) if successful_compute_delays else float(
        'inf')

    return avg_delay_of_successful_tasks, avg_compute_delay_of_successful_tasks, rewards_history
