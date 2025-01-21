# inference with NPE
from simulators import WF, GLU, SLCP
import torch
import pickle
import time
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils import BoxUniform
import argparse
import sbibm
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)


#### arguments ####
parser = argparse.ArgumentParser()
parser.add_argument('-m', "--model")
parser.add_argument('-e', "--epochs")
parser.add_argument('-n', "--num_sim")
args = parser.parse_args()

# time
start = time.time()

# Define the prior
def get_prior(sim):
    if sim == 'WF':
        prior = BoxUniform(low=torch.tensor([-2, -7, -8]), high=torch.tensor([0, -2, -2]))
    elif sim == 'GLU':
        prior = sbibm.get_task('gaussian_linear_uniform').get_prior_dist()
    elif sim == 'SLCP':
        prior = sbibm.get_task('slcp').get_prior_dist()
    else:
        raise ValueError('Unknown simulator')

    return prior

model_dict = {'GLU': GLU, 'WF': WF, 'SLCP': SLCP}

# Define the prior and simulator
sim = str(args.model)
stop_after_epochs = int(args.epochs)
prior = get_prior(sim)
simulator = model_dict[sim]
num_sim = int(args.num_sim)

# inference
# Check prior, return PyTorch prior.
prior, num_parameters, prior_returns_numpy = process_prior(prior)

# Check simulator, returns PyTorch simulator able to simulate batches.
simulator = process_simulator(simulator, prior, prior_returns_numpy)

# inference
inference = NPE(prior)
theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=num_sim)
density_estimator = inference.append_simulations(theta, x).train(stop_after_epochs=stop_after_epochs)
posterior = inference.build_posterior(density_estimator)

# Save the posterior with pickle
with open(f'{sim}/posteriors/posterior_{sim}_{num_sim}_{stop_after_epochs}.pkl', 'wb') as f:
    pickle.dump(posterior, f)

# time
end = time.time()
print(f'Inference time: {end - start} seconds')
