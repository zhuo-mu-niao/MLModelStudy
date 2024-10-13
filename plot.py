import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


base_log_dir = './result'


experiment_dirs = [
    'MobileNet+SGD+NoPretrained',
    'ResNet50+Adam+NoPretrained',
    'ResNet50+SGD+NoPretrained',
    'ResNet50+SGD+NoPretrained+10%Data',
    'ResNet50+SGD+NoPretrained+50%Data',
    'ResNet50+SGD+Pretrained',
    'VGG16+SGD+NoPretrained'
]


colors = ['orange', 'magenta', 'blue', 'green', 'grey', 'cyan', 'purple']


provided_tag = 'F1/valid'


safe_tag = provided_tag.replace('/', '_')


for i, exp_dir in enumerate(experiment_dirs):
    log_dir = os.path.join(base_log_dir, exp_dir)


    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()


    scalar_tags = event_acc.Tags().get('scalars', [])

    if provided_tag not in scalar_tags:
        print(f"Tag '{provided_tag}' not found in {exp_dir}, skipping this experiment.")
        continue


    f1_events = event_acc.Scalars(provided_tag)


    steps = [event.step for event in f1_events]
    f1_values = [event.value for event in f1_events]


    adjusted_steps = [s + 1 for s in steps]


    plt.plot(adjusted_steps, f1_values, label=exp_dir, color=colors[i % len(colors)])


plt.xticks(ticks=range(1, 21, 2))


plt.xlabel('Epochs')
plt.ylabel(f'{provided_tag}')
plt.title(f'{provided_tag}')
plt.legend()


plt.savefig(f'{safe_tag}.png')
plt.show()
