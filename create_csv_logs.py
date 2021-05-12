import json
import numpy as np
import csv, os
json_file_name = "./logs/stats.json"
with open(json_file_name) as json_file:
    data = json.load(json_file)

EPISODE_LOGFILE = 'episode_logfile'
TIMESTEP_LOGFILE = 'timestep_logfile'
directory = "./logs/timestep_logs"

TIMESTEP_FIELD_NAMES_EXTRA = ['timestep', 'is_controlled', 'vehicle_id', 'timestep_reward', 'vehicle_speed',
                            'vehicle_distance', 'mission_accomplished']

def create_timestep_log_folder(episode, vehicle_id):
    vehicle_folder_name = "vehicle_" + str(vehicle_id)

    log_folder_path = os.path.join(directory,vehicle_folder_name)

    if not os.path.exists(log_folder_path):
        os.makedirs(log_folder_path)

def get_logfile_name(log_type, **kwargs):
    vehicle_id = str(kwargs.get('vehicle_id', 0))

    assert (log_type == 'episode_average' or log_type == 'episode_mission' or
            log_type == 'episode_individual' or log_type == 'timestep'), \
        "'get_logfile_name()' only accepts 'episode_average' or 'timestep' as 'log_type'"
    logfile_name = None

    if log_type == 'episode_average':
        logfile_name = EPISODE_LOGFILE + '_average.csv'
    elif log_type == 'episode_individual':
        logfile_name = EPISODE_LOGFILE + '_individual_' + vehicle_id + '.csv'
    elif log_type == 'episode_mission':
        logfile_name = EPISODE_LOGFILE + '_mission' + '.csv'
    elif log_type == 'timestep':
        timestep = str(kwargs.get('timestep', 0))
        episode = str(kwargs.get('episode', 0))
        logfile_name = TIMESTEP_LOGFILE + '_vehicle_' + vehicle_id + '_episode_' + episode + '.csv'
        vehicle_folder_name = "vehicle_" + str(vehicle_id)
        logfile_name = os.path.join(vehicle_folder_name, logfile_name)


    logfile_path = os.path.join(directory, logfile_name)

    return logfile_path

episodes = len(data["timestamps"])

if len(data["episode_vehicle_info_debug"][0][0]) > 0:
    TIMESTEP_FIELD_NAMES_EXTRA.extend(list(data["episode_vehicle_info_debug"][0][0][0].keys()))
    TIMESTEP_FIELD_NAMES_EXTRA.extend(list(data["episode_reward_info"][0][0][0].keys()))

for episode in range(0, episodes):
    episode_length = data["episode_lengths"][episode]
    for step in range(episode_length):
        vehicle_ids = data['episode_vehicle_ids'][episode][step]
        rewards_at_timestep=[0,0,0,0]
        # for rew in range(len(data['episode_reward_info'][episode][step])):
        #     rewards_at_timestep.append(data['episode_reward_info'][episode][step][rew])
        # rewards_at_timestep = data['episode_rewards_'][episode][step]
        timestep = data['episode_timestep'][episode][step]
        for i, vehicle_id in enumerate(vehicle_ids):
            create_timestep_log_folder(episode, vehicle_id)

            vehicle_timestep_reward = 0
            if vehicle_id in data['episode_reward_ids'][episode][step]:
                j = np.where(np.array(data['episode_reward_ids'][episode][step]) == vehicle_id)[0][0]
                vehicle_timestep_reward = rewards_at_timestep[j]

            individual_timestep_log = {
                'timestep': timestep,
                'is_controlled': data['episode_vehicle_is_controlled'][episode][step][i],
                'vehicle_id': vehicle_id,
                'timestep_reward': vehicle_timestep_reward,
                'vehicle_speed': data['episode_vehicle_speeds'][episode][step][i],
                'vehicle_distance': data['episode_vehicle_distances'][episode][step][i],
                'mission_accomplished': data['episode_mission_accomplished'][episode][step]}

            individual_timestep_log_name = get_logfile_name('timestep', episode=episode,
                                                                 vehicle_id=vehicle_id, timestep=timestep)
            with open(individual_timestep_log_name, 'a') as csvfile:
                if vehicle_id in data['episode_reward_ids'][episode][step]:
                    # writer = csv.DictWriter(csvfile, fieldnames=self.TIMESTEP_FIELD_NAMES_CONTROLLED)
                    j = data['episode_reward_ids'][episode][step].index(vehicle_id)
                    individual_timestep_log.update(data["episode_reward_info"][episode][step][j])

                if len(data["episode_vehicle_info_debug"][episode][step][0]) > 0:
                    individual_timestep_log.update(data["episode_vehicle_info_debug"][episode][step][i])
                # else:
                #     writer = csv.DictWriter(csvfile, fieldnames=self.TIMESTEP_FIELD_NAMES)
                writer = csv.DictWriter(csvfile, fieldnames=TIMESTEP_FIELD_NAMES_EXTRA)
                if step == 0:
                    writer.writeheader()
                writer.writerow(individual_timestep_log)