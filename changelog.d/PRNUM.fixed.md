compute_mixed_s_matrix with the default flux magnitudes refuses a lumped or wire port on a magnetic (PMC) face,
naming the port and the face; it used to fail with an out-of-domain flux-monitor error. Such a port stands for the
full symmetric object while the flux box sees only the physical half. Use magnitude_channel='wave' or move the port off the face.
