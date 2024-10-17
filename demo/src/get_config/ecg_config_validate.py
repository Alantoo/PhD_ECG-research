from configparser import ConfigParser
from pathlib import Path


class ECGConfigException(Exception):
    pass

class ECGConfigConfig(ConfigParser):
    def __init__(self, config_file, config_block):
        super(ECGConfigConfig, self).__init__()

        if not Path(config_file).is_file():
            raise ECGConfigException(
                    'The config file %s does not exist' % config_file)
                    
        self.read(config_file)
        self.validate_config(config_block)

    def validate_config(self, config_block):
        required_values = {
            'DEFAULT': {
                'data_path' : None,
                'xls_data_path' : None,
                'fr_path' : None,
                'img_path' : None,
                'fr_img_path' : None
            },
            '%s' % (config_block): {
                'sig_name' : None,
                'file_name': None,
                'multiplier': None,
                'pathology': None,
                'data_type': ('xlsx', 'physionet', 'mat')
            }
        }

        for section, keys in required_values.items():
            if section not in self:
                raise ECGConfigException(
                    'Missing section "%s" in the config file' % section)

            for key, values in keys.items():
                if key not in self[section] or self[section][key] == '':
                    raise ECGConfigException((
                        'Missing value for "%s" under section "%s" in ' +
                        'the config file') % (key, section))

                if values:
                    if self[section][key] not in values:
                        raise ECGConfigException((
                            'Invalid value for "%s" under section "%s" in ' +
                            'the config file') % (key, section))