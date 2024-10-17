from loguru import logger
import sys

from get_config.ecg_config_validate import ECGConfigConfig, ECGConfigException

class ECGConfig:

    CONFIG_FILE_PATH = "src/config/ecg.config"

    def __init__(self, config_block, config_file_path=None):
        if config_file_path is not None:
            self.CONFIG_FILE_PATH = config_file_path
        logger.info("Read config file: {}", self.CONFIG_FILE_PATH)
        logger.debug("Config: {}", config_block)

        config = {}
        try:
            config = ECGConfigConfig(self.CONFIG_FILE_PATH, config_block)
        except ECGConfigException as e:
            logger.error("Invalid config file: {}", e)
            sys.exit(1)
    
        self.sig_name = int(config[config_block]["sig_name"].strip())
        self.pathology = int(config[config_block]["pathology"].strip())
        self.file_name = config[config_block]["file_name"].strip()
        self.multiplier = float(config[config_block]["multiplier"].strip())
        self.data_type = config[config_block]["data_type"].strip()
        self.config_block = config_block
        self.data_path = config["DEFAULT"]["data_path"].strip()
        self.xls_data_path = config["DEFAULT"]["xls_data_path"].strip()
        self.fr_path = config["DEFAULT"]["fr_path"].strip()
        self.img_path = config["DEFAULT"]["img_path"].strip()
        self.fr_img_path = config["DEFAULT"]["fr_img_path"].strip()

    def getDataType(self):
        return self.data_type

    def getPathology(self):
        return self.pathology

    def getSigName(self):
        return self.sig_name
    
    def getFileName(self):
        return self.file_name
    
    def getMultiplier(self):
        return self.multiplier
    
    def getConfigBlock(self):
        return self.config_block
    
    def getDataPath(self):
        return self.data_path
    
    def getXLSPath(self):
        return self.xls_data_path
    
    def getFrPath(self):
        return self.fr_path
    
    def getImgPath(self):
        return self.img_path
    
    def getFrImgPath(self):
        return self.fr_img_path

    def __str__(self):
        logger.debug("toString {}", self.config_block)
        return ((
                "\nConfig block: {0}\n" +
                "Signal name: {1}\n" +
                "File mame: {2}\n" +
                "Multiplier: {3}\n" +
                "Pathology: {4}\n" +
                "Default data path {5}\n" +
                "Default XLS data path {9}\n" +
                "Default fr path: {6}\n" +
                "Default images path {7}\n" +
                "Default fr images path: {8}\n"
                ).format(self.config_block,  self.sig_name, self.file_name, self.multiplier, self.pathology, self.data_path, self.fr_path, self.img_path, self.fr_img_path, self.xls_data_path))