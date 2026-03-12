import numpy as np
from joblib import Parallel, delayed


def serial_emulator_inference(emulator, X, y, v):
    y[:] ,v[:] = emulator.predict(X)

def parallel_inference(emulators, X, n_jobs=1):
    n_samples = X.shape[0]
    output_dim = len(emulators)
    M = np.zeros((n_samples, output_dim))
    V = np.zeros((n_samples, output_dim))

    Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(serial_emulator_inference)(em, X, M[:, i], V[:, i])
        for i, em in enumerate(emulators)
    )
    return M, V