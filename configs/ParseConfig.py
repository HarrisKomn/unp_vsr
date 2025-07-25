import json
import pathlib
#import DEFAULT_CONFIG_DICT, DEFAULT_CONFIG_NESTED_DICT

# def get_default_config_dict():
#
# def get_default_config_nested_dict():


def parse_config(file_path):
    # Load config
    try:
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
    except FileNotFoundError:
        print("Configuration file not found at given path '{}'".format(file_path))
        exit(1)
    # Fill in correct paths
    # config_path = pathlib.Path(file_path).parent
    # with open(config_path / 'path_info.json', 'r') as f:
    #     path_info = json.load(f)  # Dict: keys are user codes, values are a list of 'data_path', 'log_path' (absolute)
    # if user in path_info:
    #     config_dict.update({
    #         'data_path': path_info[user][0],
    #         'log_path': path_info[user][1],
    #         'ss_pretrained_path': path_info['ss_pretrained_{}'.format(user)][0]
    #     })
    # else:
    #     ValueError("User '{}' not found in configs/path_info.json".format(user))
    # # Fill in GPU device if applicable
    # if device >= 0:  # Only update config if user entered a device (default otherwise -1)
    #     config_dict['gpu_device'] = device
    #
    # # Make sure all necessary default values exist
    # default_dict = DEFAULT_CONFIG_DICT.copy()
    # default_dict.update(config_dict)  # Keeps all default values not overwritten by the passed config
    # nested_default_dicts = DEFAULT_CONFIG_NESTED_DICT.copy()
    # for k, v in nested_default_dicts.items():  # Go through the nested dicts, set as default first, then update
    #     default_dict[k] = v  # reset to default values
    #     default_dict[k].update(config_dict[k])  # Overwrite defaults with the passed config values
    #
    # # Extra config bits needed
    # default_dict['data']['transform_values']['experiment'] = default_dict['data']['experiment']

    return config_dict