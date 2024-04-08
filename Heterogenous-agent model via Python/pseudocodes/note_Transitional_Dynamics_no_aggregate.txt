Transition dynamics to steady state, no aggregate uncertainty
…
f_ss = steady_distribution
f_init = initial_distribution(t=1)
Va_ss, a_ss, c_ss = steady_state_policy
T = 5000
K_init = Aggregator (f_init, a_space)
K_guess_path = linear(Kmin, Kmax, T)
For i in max_iteration_steps:
	wage_path, rent_path = FirmFOC(K_guess_path)
	f_path, Va_path, a_path, c_path = Matrix(T, ne, na)
	Va_path[-1], a_path[-1], c_path[-1] = Va_ss, a_ss, c_ss
	f_path[1] = f_init
	For t from T-1 to 1:
		w = wage_path[t]
		r = rent_path[t]
		Va_path[t], a_path[t], c_path[t] = backward_iteration(Va_path[t+1])
	For t from 2 to T:
		f_path[t] = forward_iteration(f_path[t-1])
	K_guess_path_new = Aggregator_whole_path(f_path, a_space)
	max_error = max(absolute( K_guess_path_new - K_guess_path )
	if max_error < 1E-7:
		Break
	else:
		v = 0.2
		K_guess_path = v × K_guess_path_new + (1-v) × K_guess_path
