#=
    state = -1, 0 or 1
    -1 : random lattice of 0's and 1's
    0 : all zero s
    1 : all ones
=#
function initial_lattice_state(L, state)
    # square lattice
    lattice = Matrix{Int}(undef, L, L)
    if state == 0
        lattice = rand((-1, 1), L, L)
    else
        fill!(lattice, state)
    end
    return lattice
end
function energy(lattice, J)
    L = size(lattice, 1)
    E = 0.0
    for i in 1:L
        for j in 1:L
            E -= J * lattice[i, j] * (lattice[i, mod1(j+1, L)] + lattice[mod1(i + 1, L), j])
        end
    end
    return E
end
function metropolis_sweep!(lattice, E, M, beta, J)
    L = size(lattice, 1)
    accepted_flips = 0
    for _ in 1:L^2
        i, j = Tuple((rand(1:L), rand(1:L)))
        dE = 2 * lattice[i, j] * (
                    lattice[i, mod1(j-1, L)] +
                    lattice[i, mod1(j+1, L)] +
                    lattice[mod1(i-1, L), j] +
                    lattice[mod1(i+1, L), j])
        # acceptance criterion
        if dE <= 0 || rand() < exp(- beta * dE)
            lattice[i, j] = -lattice[i, j]
            E += dE
            M += -2 * lattice[i, j]
            accepted_flips += 1
        end
    end
    # @show accepted_flips
    return E, M
end

function compute_equilibrium_properties(lattice, T, thermal_sweeps, sampling_sweeps)
    J = 1.0
    L = size(lattice, 1)
    N = L^2
    beta = 1.0 / T

    E = energy(lattice, 1.0)
    M = sum(lattice)

    # Run metropolis sweeps to reach thermalized states.
    for _ in 1:thermal_sweeps
        E, M = metropolis_sweep!(lattice, E, M, beta, J)
    end

    E_sum = 0.0
    E2_sum = 0.0
    M_abs_sum = 0.0
    M2_sum = 0.0

    # reached thermaliza
    for _ in 1:sampling_sweeps
        E, M = metropolis_sweep!(lattice, E, M, beta, J)
        E_sum += E
        E2_sum += E^2
        M_abs_sum += abs(M)
        M2_sum += M^2
    end

    avg_E = (E_sum / sampling_sweeps) / N
    avg_M = (M_abs_sum / sampling_sweeps) / N
    Cv = (beta^2 * ((E2_sum / sampling_sweeps) - (E_sum / sampling_sweeps)^2)) / N
    chi = (beta * ((M2_sum / sampling_sweeps) - (M_abs_sum / sampling_sweeps)^2)) / N

    return avg_E, avg_M, Cv, chi
end
