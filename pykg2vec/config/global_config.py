import shutil, tarfile, urllib.request
from pathlib import Path
import os

# TODO: to be moved to utils

def extract(tar_path, extract_path='.'):
    tar = tarfile.open(tar_path, 'r')
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract(item.name, "./" + item.name[:item.name.rfind('/')])


class FreebaseFB15k(object):

    def __init__(self):
        self.name = "FB15k"
        self.url = "https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz"
        self.dataset_home_path = Path('..') / 'dataset'
        self.dataset_home_path.mkdir(parents=True, exist_ok=True)
        self.dataset_home_path = self.dataset_home_path.resolve()
        self.root_path = self.dataset_home_path / 'Freebase'
        self.tar = self.root_path / 'FB15k.tgz'
        self.downloaded_path = self.root_path / 'FB15k' / 'freebase_mtr100_mte100-'
        self.prepared_data_path = self.root_path / 'FB15k' / 'FB15k_'
        self.entity2idx_path = self.root_path / 'FB15k' / 'FB15k_entity2idx.pkl'
        self.idx2entity_path = self.root_path / 'FB15k' / 'FB15k_idx2entity.pkl'
        self.relation2idx_path = self.root_path / 'FB15k' / 'FB15k_relation2idx.pkl'
        self.idx2relation_path = self.root_path / 'FB15k' / 'FB15k_idx2relation.pkl'

        if not self.root_path.exists():
            self.download()
            self.extract()

    def download(self):
        ''' download Freebase 15k dataset from url'''
        print("Downloading the dataset %s" % self.name)

        self.root_path.mkdir()
        with urllib.request.urlopen(self.url) as response, open(str(self.tar), 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

    def extract(self):
        ''' extract the downloaded tar under Freebase 15k folder'''
        print("Extracting the downloaded dataset from %s to %s" % (self.tar, self.root_path))

        try:
            extract(str(self.tar), str(self.root_path))
        except Exception as e:
            print("Could not extract the tgz file!")
            print(type(e), e.args)

    def dump(self):
        for key, value in self.__dict__.items():
            print(key, value)


class GlobalConfig(object):

    def __init__(self, dataset='Freebase15k', negative_sample='uniform'):
        if dataset == 'Freebase15k':
            self.dataset = FreebaseFB15k()
        else:
            raise NotImplementedError("%s dataset config not found!" % dataset)

        self.negative_sample = negative_sample
        self.tmp_data = Path('..') / 'data'
        if not os.path.exists(self.tmp_data):
            os.mkdir(self.tmp_data)

    def dump(self):
        for key, value in self.dataset.__dict__.items():
            print(key, value)


class GeneratorConfig(object):
    """Configuration for Generator

        Args:
          batch_size: int value for batch size
          loss_type : string 'distance' or 'entropy' to show
                     type of algorithm used
          data_path : path from where data is read and batch is
                     generated
          sampling  :  type of sampling used for generating negative
                     triples
          queue_size : size of the generator queue
          process_num: total number of processes for retrieving and
                       preparing the data
          total_data: total number of data stored in the disk, returned
                      by data_handler
    """

    def __init__(self, batch_size=128,
                 loss_type='entropy',
                 data_path=Path('..') / 'data',
                 sampling=None,
                 queue_size=40,
                 thread_num=20,
                 data='train', 
                 algo ='ConvE'
                 ):
        self.algo = algo
        self.data = data
        self.thread_num = thread_num
        self.queue_size = queue_size
        self.sampling = sampling
        self.data_path = data_path
        self.loss_type = loss_type
        self.batch_size = batch_size



# valid_keys = ['batch_size', 'loss_type',
#               'data_path', 'sampling',
#               'queue_size', 'process_num',
#               'total_data']
# for key in valid_keys:
#     self.__dict__[key] = kwargs.get(key)


def test_init_database():
    global_config = GlobalConfig('Freebase15k')
    # global_config = GlobalConfig('Freebase100k')


if __name__ == "__main__":
    test_init_database()
