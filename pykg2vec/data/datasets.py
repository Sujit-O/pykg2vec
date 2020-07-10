import shutil
import tarfile
import pickle
import os
import zipfile
import urllib.request
from pathlib import Path
from pykg2vec.utils.logger import Logger


def extract_tar(tar_path, extract_path='.'):
    """This function extracts the tar file.

        Most of the knowledge graph datasets are downloaded in a compressed
        tar format. This function is used to extract them

        Args:
            tar_path (str): Location of the tar folder.
            extract_path (str): Path where the files will be decompressed.

        Todo:
            * Move this module to utils!
    """
    tar = tarfile.open(tar_path, 'r')
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract_tar(item.name, "./" + item.name[:item.name.rfind('/')])


def extract_zip(zip_path, extract_path='.'):
    """This function extracts the zip file.

        Most of the knowledge graph datasets are downloaded in a compressed
        zip format. This function is used to extract them

        Args:
            zip_path (str): Location of the zip folder.
            extract_path (str): Path where the files will be decompressed.

        Todo:
            * Move this module to utils!
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)


class KnownDataset:
    """The class consists of modules to handle the known datasets.

       There are various known knowledge graph datasets used by the research
       community. These datasets maybe in different format. This module
       helps in parsing those known datasets for training and testing
       the algorithms.

       Args:
          name (str): Name of the datasets
          url (str): The full url where the dataset resides.
          prefix (str): The prefix of the dataset given the website.

       Attributes:
           dataset_home_path (object): Path object where the data will be downloaded
           root_oath (object): Path object for the specific dataset.

       Examples:
           >>> from pykg2vec.data.kgcontroller import KnownDataset
           >>> name = "dL50a"
           >>> url = "https://github.com/louisccc/KGppler/raw/master/datasets/dL50a.tgz"
           >>> prefix = 'deeplearning_dataset_50arch-'
           >>> kgdata =  KnownDataset(name, url, prefix)
           >>> kgdata.download()
           >>> kgdata.extract()
           >>> kgdata.dump()

    """
    _logger = Logger().get_logger(__name__)

    def __init__(self, name, url, prefix):

        self.name = name
        self.url = url
        self.prefix = prefix

        self.dataset_home_path = Path('..') / 'dataset'
        self.dataset_home_path.mkdir(parents=True, exist_ok=True)
        self.dataset_home_path = self.dataset_home_path.resolve()
        self.root_path = self.dataset_home_path / self.name
        self.tar = self.root_path / ('%s.tgz' % self.name)
        self.zip = self.root_path / ('%s.zip' % self.name)

        if not self.root_path.exists():
            self.download()
            self.extract()

        path_eq_root = ['YAGO3_10', 'WN18RR', 'FB15K_237', 'Kinship',
                        'Nations', 'UMLS', 'NELL_995']
        if self.name == 'WN18':
            self.dataset_path = self.root_path / 'wordnet-mlj12'
        elif self.name in path_eq_root:
            self.dataset_path = self.root_path
        else:
            self.dataset_path = self.root_path / self.name

        self.data_paths = {
            'train': self.dataset_path / ('%strain.txt' % self.prefix),
            'test': self.dataset_path / ('%stest.txt' % self.prefix),
            'valid': self.dataset_path / ('%svalid.txt' % self.prefix)
        }

        self.cache_triplet_paths = {
            'train': self.dataset_path / 'triplets_train.pkl',
            'test': self.dataset_path / 'triplets_test.pkl',
            'valid': self.dataset_path / 'triplets_valid.pkl'
        }

        self.cache_metadata_path = self.dataset_path / 'metadata.pkl'
        self.cache_hr_t_path = self.dataset_path / 'hr_t.pkl'
        self.cache_tr_h_path = self.dataset_path / 'tr_h.pkl'
        self.cache_hr_t_train_path = self.dataset_path / 'hr_t_train.pkl'
        self.cache_tr_h_train_path = self.dataset_path / 'tr_h_train.pkl'
        self.cache_idx2entity_path = self.dataset_path / 'idx2entity.pkl'
        self.cache_idx2relation_path = self.dataset_path / 'idx2relation.pkl'
        self.cache_entity2idx_path = self.dataset_path / 'entity2idx.pkl'
        self.cache_relation2idx_path = self.dataset_path / 'relation2idx.pkl'
        self.cache_relationproperty_path = self.dataset_path / 'relationproperty.pkl'

    def download(self):
        ''' Downloads the given dataset from url'''
        self._logger.info("Downloading the dataset %s" % self.name)

        self.root_path.mkdir()
        if self.url.endswith('.tar.gz') or self.url.endswith('.tgz'):
            with urllib.request.urlopen(self.url) as response, open(str(self.tar), 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        elif self.url.endswith('.zip'):
            with urllib.request.urlopen(self.url) as response, open(str(self.zip), 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        else:
            raise NotImplementedError("Unknown compression format")

    def extract(self):
        ''' Extract the downloaded file under the folder with the given dataset name'''

        try:
            if os.path.exists(self.tar):
                self._logger.info("Extracting the downloaded dataset from %s to %s" % (self.tar, self.root_path))
                extract_tar(str(self.tar), str(self.root_path))
                return
            if os.path.exists(self.zip):
                self._logger.info("Extracting the downloaded dataset from %s to %s" % (self.zip, self.root_path))
                extract_zip(str(self.zip), str(self.root_path))
                return
        except Exception as e:
            self._logger.error("Could not extract the target file!")
            self._logger.exception(e)
            raise

    def read_metadata(self):
        ''' Reads the metadata of the knowledge graph if available'''
        with open(str(self.cache_metadata_path), 'rb') as f:
            meta = pickle.load(f)
            return meta

    def is_meta_cache_exists(self):
        ''' Checks if the metadata of the knowledge graph if available'''
        return self.cache_metadata_path.exists()

    def dump(self):
        ''' Displays all the metadata of the knowledge graph'''
        for key, value in self.__dict__.items():
            self._logger.info("%s %s" % (key, value))


class FreebaseFB15k(KnownDataset):
    """This data structure defines the necessary information for downloading Freebase dataset.

        FreebaseFB15k module inherits the KnownDataset class for processing
        the knowledge graph dataset.

        Attributes:
            name (str): Name of the datasets
            url (str): The full url where the dataset resides.
            prefix (str): The prefix of the dataset given the website.

    """
    def __init__(self):
        name = "FB15k"
        url = "https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz"
        prefix = "freebase_mtr100_mte100-"

        KnownDataset.__init__(self, name, url, prefix)


class DeepLearning50a(KnownDataset):
    """This data structure defines the necessary information for downloading DeepLearning50a dataset.

        DeepLearning50a module inherits the KnownDataset class for processing
        the knowledge graph dataset.

        Attributes:
            name (str): Name of the datasets
            url (str): The full url where the dataset resides.
            prefix (str): The prefix of the dataset given the website.

    """
    def __init__(self):
        name = "dL50a"
        url = "https://github.com/louisccc/KGppler/raw/master/datasets/dL50a.tgz"
        prefix = 'deeplearning_dataset_50arch-'

        KnownDataset.__init__(self, name, url, prefix)


class WordNet18(KnownDataset):
    """This data structure defines the necessary information for downloading WordNet18 dataset.

        WordNet18 module inherits the KnownDataset class for processing
        the knowledge graph dataset.

        Attributes:
            name (str): Name of the datasets
            url (str): The full url where the dataset resides.
            prefix (str): The prefix of the dataset given the website.

    """
    def __init__(self):
        name = "WN18"
        url = "https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:wordnet-mlj12.tar.gz"
        prefix = 'wordnet-mlj12-'

        KnownDataset.__init__(self, name, url, prefix)


class WordNet18_RR(KnownDataset):
    """This data structure defines the necessary information for downloading WordNet18_RR dataset.

        WordNet18_RR module inherits the KnownDataset class for processing
        the knowledge graph dataset.

        Attributes:
            name (str): Name of the datasets
            url (str): The full url where the dataset resides.
            prefix (str): The prefix of the dataset given the website.

    """
    def __init__(self):
        name = "WN18RR"
        url = "https://github.com/louisccc/KGppler/raw/master/datasets/WN18RR.tar.gz"
        prefix = ''

        KnownDataset.__init__(self, name, url, prefix)


class YAGO3_10(KnownDataset):
    """This data structure defines the necessary information for downloading YAGO3_10 dataset.

        YAGO3_10 module inherits the KnownDataset class for processing
        the knowledge graph dataset.

        Attributes:
            name (str): Name of the datasets
            url (str): The full url where the dataset resides.
            prefix (str): The prefix of the dataset given the website.

    """
    def __init__(self):
        name = "YAGO3_10"
        url = "https://github.com/louisccc/KGppler/raw/master/datasets/YAGO3-10.tar.gz"
        prefix = ''

        KnownDataset.__init__(self, name, url, prefix)


class FreebaseFB15k_237(KnownDataset):
    """This data structure defines the necessary information for downloading FB15k-237 dataset.

        FB15k-237 module inherits the KnownDataset class for processing
        the knowledge graph dataset.

        Attributes:
            name (str): Name of the datasets
            url (str): The full url where the dataset resides.
            prefix (str): The prefix of the dataset given the website.

    """
    def __init__(self):
        name = "FB15K_237"
        url = "https://github.com/louisccc/KGppler/raw/master/datasets/fb15k-237.tgz"
        prefix = ''

        KnownDataset.__init__(self, name, url, prefix)


class Kinship(KnownDataset):
    """This data structure defines the necessary information for downloading Kinship dataset.

        Kinship module inherits the KnownDataset class for processing
        the knowledge graph dataset.

        Attributes:
            name (str): Name of the datasets
            url (str): The full url where the dataset resides.
            prefix (str): The prefix of the dataset given the website.

    """
    def __init__(self):
        name = "Kinship"
        url = "https://github.com/louisccc/KGppler/raw/master/datasets/kinship.tar.gz"
        prefix = ''

        KnownDataset.__init__(self, name, url, prefix)


class Nations(KnownDataset):
    """This data structure defines the necessary information for downloading Nations dataset.

        Nations module inherits the KnownDataset class for processing
        the knowledge graph dataset.

        Attributes:
            name (str): Name of the datasets
            url (str): The full url where the dataset resides.
            prefix (str): The prefix of the dataset given the website.

    """
    def __init__(self):
        name = "Nations"
        url = "https://github.com/louisccc/KGppler/raw/master/datasets/nations.tar.gz"
        prefix = ''

        KnownDataset.__init__(self, name, url, prefix)


class UMLS(KnownDataset):
    """This data structure defines the necessary information for downloading UMLS dataset.

        UMLS module inherits the KnownDataset class for processing
        the knowledge graph dataset.

        Attributes:
            name (str): Name of the datasets
            url (str): The full url where the dataset resides.
            prefix (str): The prefix of the dataset given the website.

    """
    def __init__(self):
        name = "UMLS"
        url = "https://github.com/louisccc/KGppler/raw/master/datasets/umls.tar.gz"
        prefix = ''

        KnownDataset.__init__(self, name, url, prefix)


class NELL_995(KnownDataset):
    """This data structure defines the necessary information for downloading NELL-995 dataset.

        NELL-995 module inherits the KnownDataset class for processing
        the knowledge graph dataset.

        Attributes:
            name (str): Name of the datasets
            url (str): The full url where the dataset resides.
            prefix (str): The prefix of the dataset given the website.

    """
    def __init__(self):
        name = "NELL_995"
        url = "https://github.com/louisccc/KGppler/raw/master/datasets/NELL_995.zip"
        prefix = ''

        KnownDataset.__init__(self, name, url, prefix)


class UserDefinedDataset():
    """The class consists of modules to handle the user defined datasets.

      User may define their own datasets to be processed with the
      pykg2vec library.

      Args:
         name (str): Name of the datasets

      Attributes:
          dataset_home_path (object): Path object where the data will be downloaded
          root_oath (object): Path object for the specific dataset.

    """
    _logger = Logger().get_logger(__name__)

    def __init__(self, name, custom_dataset_path):
        self.name = name

        self.dataset_path = Path(custom_dataset_path).resolve()
        self.root_path = self.dataset_path

        if not self.root_path.exists():
            raise NotImplementedError("%s user defined dataset not found!" % self.root_path)

        train_file = self.root_path / (name + '-train.txt')
        test_file = self.root_path / (name + '-test.txt')
        valid_file = self.root_path / (name + '-valid.txt')

        if not train_file.exists():
            raise NotImplementedError("%s training file not found!" % train_file)
        if not test_file.exists():
            raise NotImplementedError("%s test file not found!" % test_file)
        if not test_file.exists():
            raise NotImplementedError("%s validation file not found!" % valid_file)

        self.data_paths = {
            'train': self.root_path / (name + '-train.txt'),
            'test': self.root_path / (name + '-test.txt'),
            'valid': self.root_path / (name + '-valid.txt')
        }

        self.cache_triplet_paths = {
            'train': self.root_path / 'triplets_train.pkl',
            'test': self.root_path / 'triplets_test.pkl',
            'valid': self.root_path / 'triplets_valid.pkl'
        }

        self.cache_metadata_path = self.root_path / 'metadata.pkl'
        self.cache_hr_t_path = self.root_path / 'hr_t.pkl'
        self.cache_tr_h_path = self.root_path / 'tr_h.pkl'
        self.cache_hr_t_train_path = self.root_path / 'hr_t_train.pkl'
        self.cache_tr_h_train_path = self.root_path / 'tr_h_train.pkl'
        self.cache_idx2entity_path = self.root_path / 'idx2entity.pkl'
        self.cache_idx2relation_path = self.root_path / 'idx2relation.pkl'
        self.cache_entity2idx_path = self.root_path / 'entity2idx.pkl'
        self.cache_relation2idx_path = self.root_path / 'relation2idx.pkl'
        self.cache_relationproperty_path = self.root_path / 'relationproperty.pkl'

    def is_meta_cache_exists(self):
        """ Checks if the metadata has been cached"""
        return self.cache_metadata_path.exists()

    def read_metadata(self):
        """ Reads the metadata of the user defined dataset"""
        with open(str(self.cache_metadata_path), 'rb') as f:
            meta = pickle.load(f)
            return meta

    def dump(self):
        """ Prints the metadata of the user-defined dataset."""
        for key, value in self.__dict__.items():
            self._logger.info("%s %s" % (key, value))
