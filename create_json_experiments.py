import json

data = {
    "info": "",
    "additional_folder_name": "",
    "base_config": "",
    "observation": {
        "observation_config":{}
    },
    "action": {
        "action_config": {}
    },
    "cruising_vehicle_front":{},
    "reward": {},
    "scenario": {},
    "merging_vehicle": {},
    "agent_config": {
        "exploration": {},
        "model": {},
        "optimizer":{}

    }
}

# ini_folder="./scripts/rl_agents_scripts/configs/experiments/IEEE_Access/"
# ini_folder = "./scripts/rl_agents_scripts/configs/experiments/IROS/"
ini_folder = "./scripts/configs/experiments/complex/"
# base_name_json = "exp_merge_IROS_"
base_name_json = "exp_merge_complex_"
start_num =400
play_with_rewards = False
play_with_merging_position = False
play_with_coop = False
play_with_mission_is_controlled = False
play_with_randomness = False
ablation_action_history = False
ablation_state_representation_f1 = False
ablation_state_representation_f2 = False
complex_scenario = False
state_representation_conv = False
IROS_exp2000 = False
IROS_exp3000 = False
IROS_exp4000 = False
IROS_exp5000 = False
IROS_exp9100 = False
complex_100 = False
complex_400 = True

if complex_100:
    base_config_folder = "configs/experiments/complex/"
    start_num = 100

    for base_config in ["exp_merge_complex_100base-1.json","exp_merge_complex_100base-2.json", "exp_merge_complex_100base-3.json",  "exp_merge_complex_100base-3-2.json"]:
        for controlled_vehicle in [True, False]:
            data["base_config"] = base_config_folder + base_config
            data["merging_vehicle"]["controlled_vehicle"] = controlled_vehicle

            info = "exp_merge_" + str(start_num) + " similar to " + data["base_config"] + \
                   " controlled_vehicle " + str(controlled_vehicle)

            data["additional_folder_name"] = "exp_merge_complex" + str(start_num)
            data['info'] = info
            data["tracker_logging"] = True

            name = ini_folder + base_name_json + str(start_num) + '.json'
            start_num += 1
            with open(name, 'w') as outfile:
                json.dump(data, outfile)

if complex_400:
    base_config_folder = "configs/experiments/complex/"
    start_num = 410
    for base_config in ["exp_merge_complex_400base-1.json","exp_merge_complex_400base-2.json","exp_merge_complex_400base-3.json",
                        "exp_merge_complex_400base-4.json","exp_merge_complex_400base-5.json","exp_merge_complex_400base-6.json"]:
        data["base_config"] = base_config_folder + base_config
        data["reward"]["cooperative_flag"] = True
        data["reward"]["sympathy_flag"] = True
        for enable_lane_change in [True, False]:

            data["cruising_vehicle_front"]["enable_lane_change"] = enable_lane_change

            for lateral in [True, False]:

                data["action"]["action_config"]["lateral"] = lateral
                info = "exp_merge_" + str(start_num) + " similar to " + data["base_config"] + \
                       " enable_lane_change " + str(enable_lane_change) + \
                       " lateral " + str(lateral)

                data["additional_folder_name"] = "exp_merge_complex" + str(start_num)
                data['info'] = info
                data["tracker_logging"] = True

                name = ini_folder + base_name_json + str(start_num) + '.json'
                start_num += 1
                with open(name, 'w') as outfile:
                    json.dump(data, outfile)



if IROS_exp9100:
    base_config_folder = "configs/experiments/IROS/"

    for base_config in ["exp_merge_IROS_9000base-1.json", "exp_merge_IROS_9000base-2.json",
                        "exp_merge_IROS_9000base-2-1.json","exp_merge_IROS_9000base-3.json",
                        "exp_merge_IROS_9000base-3-1.json","exp_merge_IROS_9000base-3-2.json"]:
        for random_offset_merging in [20]:
            data["base_config"] = base_config_folder + base_config
            data["merging_vehicle"]["random_offset_merging"] = [-random_offset_merging , random_offset_merging]

            info = "exp_merge_" + str(start_num) + " similar to " + data["base_config"] + \
                   " random_offset_merging " + str(random_offset_merging)

            data["additional_folder_name"] = "exp_merge_IROS" + str(start_num)
            data['info'] = info
            data["tracker_logging"] = True

            name = ini_folder + base_name_json + str(start_num) + '.json'
            start_num += 1
            with open(name, 'w') as outfile:
                json.dump(data, outfile)
if IROS_exp2000:
    base_config_folder = "configs/experiments/IROS/"
    for network in ["ConvNetStanfordMARLNoRes"]:
        for base_config in ["exp_merge_IROS_2000_base-1.json", "exp_merge_IROS_2000_base-2.json",
                            "exp_merge_IROS_2000_base-3.json"]:
            for gama in [0.8, 0.99]:
                for memory_capacity in [20000]:
                    for target_update in [50, 500]:
                        for tau in [10000]:
                            data["base_config"] = base_config_folder + base_config
                            data["agent_config"]["gamma"] = gama
                            data["agent_config"]["memory_capacity"] = memory_capacity
                            data["agent_config"]["target_update"] = target_update
                            data["agent_config"]["exploration"]["tau"] = tau
                            data["agent_config"]["model"]["type"] = network

                            info = "exp_merge_" + str(start_num) + " similar to " + data["base_config"] + \
                                   " model " + str(network) + \
                                   " tau " + str(tau) + \
                                   " target_update " + str(target_update) + \
                                   " memory_capacity " + str(memory_capacity) + \
                                   " gama " + str(gama)

                            data["additional_folder_name"] = "exp_merge_IROS" + str(start_num)
                            data['info'] = info
                            data["tracker_logging"] = True

                            name = ini_folder + base_name_json + str(start_num) + '.json'
                            start_num += 1
                            with open(name, 'w') as outfile:
                                json.dump(data, outfile)
if IROS_exp3000:
    base_config_folder = "configs/experiments/IROS/"
    for network in ["ConvNet3Layer"]:
        for base_config in ["exp_merge_IROS_3000base-1.json", "exp_merge_IROS_3000base-2.json",
                            "exp_merge_IROS_3000base-3.json","exp_merge_IROS_3000base-4.json"]:
                for double in [False]:
                    for optimizer in ["ADAM", "RMS_PROP"]:
                        for policy_frequency in [1, 5]:
                            data["agent_config"]["model"]["type"] = network
                            data["base_config"] = base_config_folder + base_config
                            data["agent_config"]["double"] = double
                            data["agent_config"]["optimizer"]["type"] = optimizer
                            data["policy_frequency"] = policy_frequency
                            info = "exp_merge_" + str(start_num) + " similar to " + data["base_config"] + \
                                   " double " + str(double) + \
                                   " optimizer " + str(optimizer) + \
                                   " network " + str(network) + \
                                   " policy_frequency " + str(policy_frequency)

                            data["additional_folder_name"] = "exp_merge_IROS" + str(start_num)
                            data['info'] = info
                            data["tracker_logging"] = True

                            name = ini_folder + base_name_json + str(start_num) + '.json'
                            start_num += 1
                            with open(name, 'w') as outfile:
                                json.dump(data, outfile)
if IROS_exp4000:
    base_config_folder = "configs/experiments/IROS/"
    for network in ["ConvNetStanfordMARLNoRes", "ConvNetStanfordMARLRes"]:
        for base_config in ["exp_merge_IROS_4000base-1.json", "exp_merge_IROS_4000base-2.json",
                            "exp_merge_IROS_4000base-3.json","exp_merge_IROS_4000base-4.json"]:
                for tau in [20000]:
                    for exploration in ["EpsilonGreedyLinear"]:
                    # for exploration in ["EpsilonGreedy", "EpsilonGreedyLinear"]:
                    #     for observation_shape in [[500, 64], [200, 64]]:
                        for observation_shape in [[200,64]]:
                            data["agent_config"]["model"]["type"] = network
                            data["base_config"] = base_config_folder + base_config
                            # data["agent_config"]["double"] = double
                            # data["agent_config"]["optimizer"]["type"] = optimizer
                            # data["policy_frequency"] = policy_frequency
                            data["agent_config"]["exploration"]["method"] = exploration
                            data["agent_config"]["exploration"]["tau"] = tau
                            data["observation"]["observation_config"]["observation_shape"] = observation_shape
                            data["observation"]["observation_config"]["observation_shape"] = observation_shape
                            info = "exp_merge_" + str(start_num) + " similar to " + data["base_config"] + \
                                   " exploration " + str(exploration) + \
                                   " tau " + str(tau) + \
                                   " observation_shape " + str(observation_shape) + \
                                   " network " + str(network)

                            data["additional_folder_name"] = "exp_merge_IROS" + str(start_num)
                            data['info'] = info
                            data["tracker_logging"] = True

                            name = ini_folder + base_name_json + str(start_num) + '.json'
                            start_num += 1
                            with open(name, 'w') as outfile:
                                json.dump(data, outfile)
if IROS_exp5000:
    base_config_folder = "configs/experiments/IROS/"
    for network in ["ConvNet3D", "ConvNet3DResidual"]:
        for base_config in ["exp_merge_IROS_5000base-1.json", "exp_merge_IROS_5000base-2.json" ]:
                for tau in [20000]:
                    for exploration in ["EpsilonGreedyLinear"]:
                    # for exploration in ["EpsilonGreedy", "EpsilonGreedyLinear"]:
                        for observation_shape in [[500, 64], [200, 64]]:
                        # for observation_shape in [[200,64]]:
                            for history_stack_size in [5,10,15]:

                                data["agent_config"]["model"]["type"] = network
                                data["base_config"] = base_config_folder + base_config
                                # data["agent_config"]["double"] = double
                                # data["agent_config"]["optimizer"]["type"] = optimizer
                                # data["policy_frequency"] = policy_frequency
                                data["agent_config"]["exploration"]["method"] = exploration
                                data["agent_config"]["exploration"]["tau"] = tau
                                data["observation"]["observation_config"]["observation_shape"] = observation_shape
                                data["observation"]["observation_config"]["observation_shape"] = observation_shape
                                data["observation"]["observation_config"]["history_stack_size"] = history_stack_size

                                info = "exp_merge_" + str(start_num) + " similar to " + data["base_config"] + \
                                       " exploration " + str(exploration) + \
                                       " tau " + str(tau) + \
                                       " observation_shape " + str(observation_shape) + \
                                       " history_stack_size " + str(history_stack_size) + \
                                       " network " + str(network)

                                data["additional_folder_name"] = "exp_merge_IROS" + str(start_num)
                                data['info'] = info
                                data["tracker_logging"] = True

                                name = ini_folder + base_name_json + str(start_num) + '.json'
                                start_num += 1
                                with open(name, 'w') as outfile:
                                    json.dump(data, outfile)

if play_with_rewards:
    for collision_reward in [0, -2]:
        for successful_merging_reward in [0,10]:
            for high_speed_reward in [1]:
                for cooperative_reward in [4]:
                    data["reward"]["cooperative_flag"] = True
                    data["reward"]["sympathy_flag"] = True

                    if successful_merging_reward == 0:
                        data["reward"]["sympathy_flag"] = False

                    info = "exp_merge_" + str(start_num) + " similar to " + data["base_config"] + \
                           " cooperative_flag " + str(data["reward"]["cooperative_flag"]) + \
                           " sympathy_flag " + str(data["reward"]["sympathy_flag"]) + \
                           " collision_reward= " + str(collision_reward) + " and " + \
                           "successful_merging_reward= " + str(successful_merging_reward) + " and " + \
                           "high_speed_reward= " + str(high_speed_reward) + " and " + \
                           "cooperative_reward= " + str(cooperative_reward)

                    data["additional_folder_name"] = "exp_merge_" + str(start_num)
                    data['info'] = info
                    data["reward"]['collision_reward'] = collision_reward
                    data["reward"]['successful_merging_reward'] = successful_merging_reward
                    data["reward"]['high_speed_reward'] = high_speed_reward
                    data["reward"]['cooperative_reward'] = cooperative_reward
                    data["tracker_logging"] = True
                    name = ini_folder + base_name_json + str(start_num) + '.json'
                    start_num += 1
                    with open(name, 'w') as outfile:
                        json.dump(data, outfile)

if complex_scenario:
    for on_desired_lane_reward in [2, 5, 10]:
        data["reward"]["cooperative_flag"] = True
        data["reward"]["sympathy_flag"] = False
        data["additional_folder_name"] = "exp_merge_" + str(start_num)
        data["reward"]['on_desired_lane_reward'] = on_desired_lane_reward
        data["tracker_logging"] = True

        info = "exp_merge_" + str(start_num) + " similar to " + data["base_config"] + \
               " cooperative_flag " + str(data["reward"]["cooperative_flag"]) + \
               " sympathy_flag " + str(data["reward"]["sympathy_flag"]) + \
               " on_desired_lane_reward= " + str(on_desired_lane_reward)

        data['info'] = info
        name = ini_folder + "exp_merge_" + str(start_num) + '.json'
        start_num += 1
        with open(name, 'w') as outfile:
            json.dump(data, outfile)

if play_with_merging_position:
    for initial_position in [90, 95, 100]:
        for speed in [23, 25, 27]:
            data["reward"]["cooperative_flag"] = True
            data["reward"]["sympathy_flag"] = False

            info = "exp_merge_" + str(start_num) + " similar to IEEE_Access/exp_base " + \
                   " cooperative_flag " + str(data["reward"]["cooperative_flag"]) + \
                   " sympathy_flag " + str(data["reward"]["sympathy_flag"]) + \
                   "initial_position= " + str(initial_position) + " and " + \
                   "speed= " + str(speed)

            data["additional_folder_name"] = "exp_merge_" + str(start_num)
            data['info'] = info
            data["merging_vehicle"]['initial_position'] = [initial_position, 0]
            data["merging_vehicle"]['speed'] = speed

            data["tracker_logging"] = True

            name = ini_folder + "exp_merge_" + str(start_num) + '.json'
            start_num += 1
            with open(name, 'w') as outfile:
                json.dump(data, outfile)

if play_with_coop:
    base_config_folder = "configs/experiments/complex/"
    start_num = 400
    for base_config in ["exp_merge_complex_400base-1.json"]:
        data["base_config"] = base_config_folder + base_config
        for cooperative_flag, sympathy_flag in [(True, True), (True, False), (False, False)]:
            # for type in ["highway_env.vehicle.behavior.CustomVehicle" , "highway_env.vehicle.behavior.CustomVehicleAggressive"]:
            for controlled_vehicle in [True, False]:
                for type in ["highway_env.vehicle.behavior.CustomVehicle"]:
                    # data["action"]["action_config"]["lateral"] = True
                    data["reward"]["cooperative_flag"] = cooperative_flag
                    data["reward"]["sympathy_flag"] = sympathy_flag
                    # data["observation"]["cooperative_perception"] = True
                    data["additional_folder_name"] = "exp_merge_" + str(start_num)
                    data["merging_vehicle"]['vehicles_type'] = type
                    data["merging_vehicle"]['controlled_vehicle'] = controlled_vehicle # True
                    # data["merging_vehicle"]['max_speed'] = 25
                    data["tracker_logging"] = True

                    info = "exp_merge_" + str(start_num) + " similar to IEEE_Access/exp_base " + \
                           " cooperative_flag " + str(data["reward"]["cooperative_flag"]) + \
                           " sympathy_flag " + str(data["reward"]["sympathy_flag"]) + \
                           " type= " + str(type) + \
                           " controlled_vehicle " + str(data["merging_vehicle"]['controlled_vehicle'])
                    data['info'] = info

                    name = ini_folder + base_name_json + str(start_num) + '.json'
                    start_num += 1
                    with open(name, 'w') as outfile:
                        json.dump(data, outfile)

if play_with_randomness:
    for cooperative_flag, sympathy_flag in [(True, True), (True, False), (False, False)]:
        # for type in ["highway_env.vehicle.behavior.CustomVehicle" , "highway_env.vehicle.behavior.CustomVehicleAggressive"]:
        for type in ["highway_env.vehicle.behavior.CustomVehicle"]:
            for random_offset_merging, randomize_speed_offset in zip([2, 10, 15], [0, 4, 6]):
                data["reward"]["cooperative_flag"] = cooperative_flag
                data["reward"]["sympathy_flag"] = sympathy_flag
                data["observation"]["cooperative_perception"] = True
                data["additional_folder_name"] = "exp_merge_" + str(start_num)
                data["merging_vehicle"]['vehicles_type'] = type
                data["merging_vehicle"]['controlled_vehicle'] = True
                data["merging_vehicle"]['random_offset_merging'] = [-random_offset_merging, random_offset_merging]
                data["scenario"]['randomize_speed_offset'] = [-randomize_speed_offset, randomize_speed_offset]
                data["tracker_logging"] = True

                info = "exp_merge_" + str(start_num) + " similar to IEEE_Access/exp_base " + \
                       " cooperative_flag " + str(data["reward"]["cooperative_flag"]) + \
                       " sympathy_flag " + str(data["reward"]["sympathy_flag"]) + \
                       " type= " + str(type) + \
                       " cooperative_perception " + str(data["observation"]["cooperative_perception"]) + \
                       " controlled_vehicle" + str(data["merging_vehicle"]['controlled_vehicle']) + \
                       " random_offset_merging" + str(data["merging_vehicle"]['random_offset_merging']) + \
                       " randomize_speed_offset" + str(data["scenario"]['randomize_speed_offset'])

                data['info'] = info

                name = ini_folder + "exp_merge_" + str(start_num) + '.json'
                start_num += 1
                with open(name, 'w') as outfile:
                    json.dump(data, outfile)

    # for type in ["highway_env.vehicle.behavior.CustomVehicle", "highway_env.vehicle.behavior.CustomVehicleAggressive"]:
    #     data["reward"]["cooperative_flag"] = False
    #     data["reward"]["sympathy_flag"] = False
    #     data["observation"]["cooperative_perception"] = False
    #
    #     info = "exp_merge_" + str(start_num) + " similar to IEEE_Access/exp_base " + \
    #            " cooperative_flag " + str(data["reward"]["cooperative_flag"]) + \
    #            " sympathy_flag " + str(data["reward"]["sympathy_flag"]) + \
    #            " type= " + str(type) + " cooperative_perception " +  str(data["observation"]["cooperative_perception"])
    #
    #     data["additional_folder_name"] = "exp_merge_" + str(start_num)
    #     data['info'] = info
    #     data["merging_vehicle"]['vehicles_type'] = type
    #     data["tracker_logging"] = True
    #
    #     name = ini_folder + "exp_merge_" + str(start_num) + '.json'
    #     start_num += 1
    #     with open(name, 'w') as outfile:
    #         json.dump(data, outfile)

if play_with_mission_is_controlled:
    type = "ControlledVehicle"
    for cooperative_flag, sympathy_flag in [(True, True), (True, False), (False, False)]:
        data["reward"]["cooperative_flag"] = cooperative_flag
        data["reward"]["sympathy_flag"] = sympathy_flag
        data["observation"]["cooperative_perception"] = True

        info = "exp_merge_" + str(start_num) + " similar to IEEE_Access/exp_base " + \
               " cooperative_flag " + str(data["reward"]["cooperative_flag"]) + \
               " sympathy_flag " + str(data["reward"]["sympathy_flag"]) + \
               " type= " + str(type) + " cooperative_perception " + str(data["observation"]["cooperative_perception"])

        data["additional_folder_name"] = "exp_merge_" + str(start_num)
        data['info'] = info
        data["merging_vehicle"]['vehicles_type'] = type
        data["tracker_logging"] = True

        name = ini_folder + "exp_merge_" + str(start_num) + '.json'
        start_num += 1
        with open(name, 'w') as outfile:
            json.dump(data, outfile)

    data["reward"]["cooperative_flag"] = False
    data["reward"]["sympathy_flag"] = False
    data["observation"]["cooperative_perception"] = False

    info = "exp_merge_" + str(start_num) + " similar to IEEE_Access/exp_base " + \
           " cooperative_flag " + str(data["reward"]["cooperative_flag"]) + \
           " sympathy_flag " + str(data["reward"]["sympathy_flag"]) + \
           " type= " + str(type) + " cooperative_perception " + str(data["observation"]["cooperative_perception"])

    data["additional_folder_name"] = "exp_merge_" + str(start_num)
    data['info'] = info
    data["merging_vehicle"]['vehicles_type'] = type
    data["tracker_logging"] = True

    name = ini_folder + "exp_merge_" + str(start_num) + '.json'
    start_num += 1
    with open(name, 'w') as outfile:
        json.dump(data, outfile)

if ablation_action_history:
    for action_history_type in ["discrete", "xy_discrete", "binary"]:
        for action_history_count in [0, 2, 5, 10, 15]:
            data["observation"]["action_history_type"] = action_history_type
            data["observation"]["action_history_count"] = action_history_count

            info = "exp_merge_" + str(start_num) + " similar to IEEE_Access/exp_merge_300 " + \
                   " action_history_type " + str(data["observation"]["action_history_type"]) + \
                   " action_history_count " + str(data["observation"]["action_history_count"])

            data["additional_folder_name"] = "exp_merge_" + str(start_num)
            data['info'] = info
            data["tracker_logging"] = True

            name = ini_folder + "exp_merge_" + str(start_num) + '.json'
            start_num += 1
            with open(name, 'w') as outfile:
                json.dump(data, outfile)

if ablation_state_representation_f1:
    for mission_vehicle_observation in [True, False]:
        for features in [["presence", "x", "y", "vx", "vy", "is_controlled"], ["presence", "x", "y", "vx", "vy"],
                         ["x", "y", "vx", "vy", "is_controlled"]]:
            for order in ["sorted", "sorted_by_id", "shuffled"]:  # wrong shuffled sorted_by_x
                data["observation"]["mission_vehicle_observation"] = mission_vehicle_observation
                data["observation"]["features"] = features
                data["observation"]["order"] = order

                info = "exp_merge_" + str(start_num) + " similar to IEEE_Access/exp_merge_300 " + \
                       " mission_vehicle_observation " + str(data["observation"]["mission_vehicle_observation"]) + \
                       " features " + str(data["observation"]["features"]) + \
                       " order " + str(data["observation"]["order"])

                data["additional_folder_name"] = "exp_merge_" + str(start_num)
                data['info'] = info
                data["tracker_logging"] = True

                name = ini_folder + "exp_merge_" + str(start_num) + '.json'
                start_num += 1
                with open(name, 'w') as outfile:
                    json.dump(data, outfile)

if ablation_state_representation_f2:
    for absolute in [True, False]:
        for see_behind in [True, False]:
            for normalize in [True, False]:
                for cooperative_perception in [True, False]:
                    data["observation"]["absolute"] = absolute
                    data["observation"]["see_behind"] = see_behind
                    data["observation"]["normalize"] = normalize
                    data["observation"]["cooperative_perception"] = cooperative_perception

                    info = "exp_merge_" + str(start_num) + " similar to IEEE_Access/exp_merge_300 " + \
                           " absolute " + str(data["observation"]["absolute"]) + \
                           " see_behind " + str(data["observation"]["see_behind"]) + \
                           " normalize " + str(data["observation"]["normalize"]) + \
                           " cooperative_perception " + str(data["observation"]["cooperative_perception"])

                    data["additional_folder_name"] = "exp_merge_" + str(start_num)
                    data['info'] = info
                    data["tracker_logging"] = True

                    name = ini_folder + "exp_merge_" + str(start_num) + '.json'
                    start_num += 1
                    with open(name, 'w') as outfile:
                        json.dump(data, outfile)

if state_representation_conv:
    base_config_folder = "configs/experiments/IROS/"
    for cruising_vehicles_front, lanes_count in [(False, 1), (False, 3), (True, 3)]:
        for base_config in ["exp_base_conv_multiagent_grid.json", "exp_base_conv_multiagent_heatmap.json",
                            "exp_base_conv_multiagent_image.json"]:
            data["cruising_vehicles_front"] = cruising_vehicles_front
            data["cruising_vehicles_front_random_everywhere"] = cruising_vehicles_front
            data["lanes_count"] = lanes_count
            data["base_config"] = base_config_folder + base_config

            info = "exp_merge_" + str(start_num) + " similar to " + data["base_config"] + \
                   " cruising_vehicles_front " + str(data["cruising_vehicles_front"]) + \
                   " lanes_count " + str(data["lanes_count"])

            data["additional_folder_name"] = "exp_merge_IROS" + str(start_num)
            data['info'] = info
            data["tracker_logging"] = True

            name = ini_folder + base_name_json + str(start_num) + '.json'
            start_num += 1
            with open(name, 'w') as outfile:
                json.dump(data, outfile)

print("End")
