from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DIRG_DATA_DIR = ROOT_DIR / "data" / "DIRG"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"

TASK = 2
device = "cuda"

#DIRG任务划分
DIRG_TASK_DOMAINS = {
    1: {
        'src': [(100,0),(200,0),(300,0),(400,0),(100,700),(200,700),(300,700),(400,700)],  
        'tgt': [(100,500),(200,500),(300,500),(400,500)]   
    },
    2: {
        'src': [(100,0),(100,500),(100,700),(100,900),(300,0),(300,500),(300,700),(300,900)], 
        'tgt': [(200,0),(200,500),(200,700),(200,900)] 
    },
    3: {
        'src': [(100,0),(200,0),(300,0),(400,0),(100,500),(200,500),(300,500),(400,500)], 
        'tgt': [(100,700),(200,700),(300,700),(400,700)] 
    },
    4: {
        'src': [(100,0),(100,700),(300,0),(300,700)], 
        'tgt': [(200,0),(400,0),(100,500),(300,500),(200,700),(400,700),(100,900),(300,900)] 
    },
}
DIRG_task_src = DIRG_TASK_DOMAINS[TASK]['src']
DIRG_task_tgt = DIRG_TASK_DOMAINS[TASK]['tgt']

#MEDG权重
num_classes=7
epochs = 100
weight_outer = 0.3
weight_coral=0.3
weight_adv = 0.7
weight_domainacc = 0.3
weight_HSIC = 0.1
weight_rec = 0.1

#DANN0权重
DANN0_num_classes = 7
DANN0_epochs = 100
DANN0_weight_domain = 0.5
DANN0_batch_size = 64
DANN0_lr = 0.0005

#DANN权重
DANN_num_classes = 7
DANN_epochs = 100
DANN_weight_domain = 1
DANN_batch_size = 64
DANN_lr = 0.0005

#MCD权重
MCD_num_classes = 7
MCD_epochs = 100
MCD_batch_size = 64
MCD_lr = 0.0005

#域分析
pretrained_model_path = MODELS_DIR / "besttask4_99,36.pt"
Domain_num_classes = 7