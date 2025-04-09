from torch.utils.data import Dataset
from utils.dataset_utils import method_set,method_name_dct

class PassagesDataset(Dataset):
    def __init__(self, dataset,mode='checkgpt',need_ids=True):
        self.mode=mode
        self.dataset = dataset
        self.need_ids=need_ids
        self.mode_set=('checkgpt_origin','checkgpt','middle','HC3_origin','seqxgpt_origin','HC3')
        self.classes=[]
        self.method_name_set={}
                
        if mode in self.mode_set:
            cnt=0
            print(method_name_dct)
            for method_set_name,m_set in method_name_dct.items():
                for name in m_set:
                    self.method_name_set[name]=(cnt,method_set[method_set_name])
                    self.classes.append(name)
                    cnt+=1
        else:
            LLM_name=set()
            for item in self.dataset:
                LLM_name.add(item[2])
            for i,name in enumerate(LLM_name):
                self.method_name_set[name]=(i,i)
                self.classes.append(name)
        
        print(f'there are {len(self.classes)} classes in {mode} dataset')
        print(f'the classes are {self.classes}')
    
    def get_class(self):
        return self.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text,label,src,id=self.dataset[idx]
        '''
        Initialize write_model and write_model_set to 1000 as default values or error makers.
        Design intention: Override this default value through subsequent logic, and trigger an assertion error if it is not overridden.
        '''
        attack_method,attack_method_set=1000,1000
        for name in self.method_name_set.keys():
            if name in src:
                attack_method,attack_method_set=self.method_name_set[name]
                break
        assert attack_method!=1000,f'attack method not exists,src is {src}'

        if self.need_ids:
            return text,int(label),int(attack_method),int(attack_method_set),int(id)
        else:
            return text,int(label),int(attack_method),int(attack_method_set)