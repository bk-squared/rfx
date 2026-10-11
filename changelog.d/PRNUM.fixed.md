compute_mixed_s_matrix with the default flux magnitudes now refuses a lumped or wire port on a magnetic face.
Such a port stands for the full symmetric object; the flux box sees only the half.
Use magnitude_channel='wave', or move the port off the face.
