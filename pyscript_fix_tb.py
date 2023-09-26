import os 

d = os.listdir('tensorboard_old/roberta_base_mlm')
tf_recs = [x for x in d if 'tfevents' in x]
runs = list(set([x.split('.')[3] for x in tf_recs]))

for run in runs:
    if not os.path.exists(f'tensorboard_old/roberta_base_mlm/run_{run}'):
        os.makedirs(f'tensorboard_old/roberta_base_mlm/run_{run}')
    command = f'mv tensorboard_old/roberta_base_mlm/events.out.tfevents.{run}* tensorboard_old/roberta_base_mlm/run_{run}'
    print(command)
    os.system(command)

