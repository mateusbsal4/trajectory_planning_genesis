import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
import time 

# Generate training data
train_X = torch.rand(10, 40, dtype=torch.double) * 2
Y = 1 - torch.linalg.norm(train_X - 0.5, dim=-1, keepdim=True)
Y = Y + 0.1 * torch.randn_like(Y)  # add some noise

# Start timer
start = time.time()

# Define the Gaussian Process model
gp = SingleTaskGP(
    train_X=train_X,
    train_Y=Y,
    input_transform=Normalize(d=40),
    outcome_transform=Standardize(m=1),
)

# Define the marginal log likelihood
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

# Fit the model
print("start fitting")
fit_gpytorch_mll(mll)

# Define the batch acquisition function (qLogExpectedImprovement)
logEI = qLogExpectedImprovement(model=gp, best_f=Y.max())

# Define the bounds for the optimization
bounds = torch.stack([torch.zeros(40), torch.ones(40)]).to(torch.double)

# Optimize the acquisition function for q=8
candidate, acq_value = optimize_acqf(
    logEI, bounds=bounds, q=8, num_restarts=5, raw_samples=20,
)

# Print results
print("bo time: %.2f seconds" % (time.time() - start))
print("Candidate points:")
print(candidate)  # Tensor of shape (8, 40)
print("Acquisition values:")
print(acq_value)  # Tensor of shape (8,)