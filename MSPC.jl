
function loadData(model)
    if model[:dataset] == "tenessee"
        X_healthy = readdlm("Fault Detection Data/d00_te.dat")
        # Get list of fault numbers
        fault_nums = model[:dataSubset]

        # Initialize an empty array to store all datasets
        X_faulty_list = []

        # Loop through each fault number
        for fault_num in fault_nums
            if fault_num <= 9
                fault_file = "Fault Detection Data/d0$(fault_num).dat"
            else
                fault_file = "Fault Detection Data/d$(fault_num).dat"
            end
            X = readdlm(fault_file)
            push!(X_faulty_list, X)
        end
        X_faulty = vcat(X_faulty_list...)  

        test_num = model[:testFault]
        if test_num <= 9
            test_file = "Fault Detection Data/d0$(test_num)_te.dat"
        else
            test_file = "Fault Detection Data/d$(test_num)_te.dat"
        end
        X_test = readdlm(test_file) 
        
    end    

    println("Size of X_healthy before: ", size(X_healthy))
    println("Size of X_faulty before: ", size(X_faulty))
    println("Size of X_test", size(X_test))

    # Fix dimension mismatch
    if size(X_healthy, 2) != size(X_faulty, 2)
        println("Transposing X_faulty to match dimensions.")
        X_faulty = X_faulty'  # Transpose the faulty dataset
    elseif size(X_healthy, 2) != size(X_test, 2)
        X_test = X_test'  # Transpose the faulty dataset
    end

    println("Size of X_healthy after: ", size(X_healthy))
    println("Size of X_faulty after: ", size(X_faulty))

    # Ensure same number of columns
    min_cols = min(size(X_healthy, 2), size(X_faulty, 2))
    X_healthy = X_healthy[:, 1:min_cols]
    X_faulty = X_faulty[:, 1:min_cols]

    if model[:scale] == true
        model[:XHealthy], model[:muX], model[:stdX] = zscore(X_healthy)
        model[:XFaulty] = normalize(X_faulty, model[:muX], model[:stdX])
        model[:XTest]   = normalize(X_test, model[:muX], model[:stdX])
    else
        model[:XHealthy] = X_healthy
        model[:XFaulty] = X_faulty
        model[:XTest]   = X_test
    end

    np = size(model[:XTest], 1)
    model[:YTest] = vcat(zeros(160), ones(np-160))

    if model[:kernelVersion] == "individual"
        row, col = size(model[:XHealthy])
        model[:initialParams] = ones(col)
    
    elseif model[:kernelVersion] == "combined"
        model[:initialParams] = [1.0, 1.0]
 
    elseif model[:kernelVersion] == "family"
        model[:initialParams] = ones(11)
        
    elseif model[:kernelVersion] == "individualScale"
        row, col = size(model[:XHealthy])
        model[:initialParams] = ones((col*2) + 1)
    end

    return model
end

function zscore(data)
    mu = mean(data, dims=1)
    sigma = std(data, dims=1)
    return (data .- mu) ./ (sigma .+ eps()), mu, sigma  # Prevent division by zero
end

function normalize(data, mu, sigma)
    return (data .- mu) ./ (sigma .+ eps())
end


function normDiff(x, y)
    sqx = x .^ 2
    sqy = y .^ 2
    nsx = sum(sqx, dims=2)
    nsy = sum(sqy, dims=2)
    nsx = vec(nsx)
    nsy = vec(nsy)
    innerMat = x * y'
    nD = -2 .* innerMat .+ nsx .+ nsy'
    return nD
end

function kernelRBF2(X::Matrix{Float64}, params, model::Dict{Symbol,Any}, state)
    if model[:kernelVersion] == "individual"
        noRows, noVars = size(X)
        K = zeros(noRows, noRows)
        
        for i = 1:noVars
            ND = normDiff(X[:,i], X[:,i])
            if model[:kernelType] == "gaussian"
                K = K + exp.(-ND)/(2*params[i]^2)
            elseif model[:kernelType] == "matern1/2"
                row, col = size(ND)
                d = sqrt.(abs.(ND))
                K = K + exp.(-d/params[i])
            elseif model[:kernelType] == "matern3/2"
                row, col = size(ND)
                d = sqrt.(abs.(ND))
                sqrt3_d = sqrt(3) * d / params[i]
                K = K + (1. .+ sqrt3_d) .* exp.(-sqrt3_d)
            elseif model[:kernelType] == "matern5/2"
                row, col = size(ND)
                d = sqrt.(abs.(ND))
                sqrt5_d = sqrt(5)*ones(row,col) + d / params[i]
                term2 = (5.0 * ND) / (3.0 * params[i]^2)
                K = K + (1. .+ sqrt5_d .+ term2) .* exp.(-sqrt5_d)
            elseif model[:kernelType] == "cauchy"
                row, col = size(ND)
                K = K + 1. ./ (ones(row, col) + (ND ./ params[i]^2))
            else 
                println("The kernel method is not available for individual calculations.")
            end
        end
    
    elseif model[:kernelVersion] == "individualScale"
        noRows, noVars = size(X)
        K = zeros(noRows, noRows)
        
        for i = 1:noVars
            ND = normDiff(X[:,i], X[:,i])
            if model[:kernelType] == "gaussian"
                K = K + params[i + noVars] .* exp.(-ND)/(2*params[i]^2)
            elseif model[:kernelType] == "matern1/2"
                row, col = size(ND)
                d = sqrt.(ND)
                K = K + params[i + noVars] .* exp.(-d/params[i])
            elseif model[:kernelType] == "matern3/2"
                row, col = size(ND)
                d = sqrt.(ND + 1e-8.*ones(row,col))
                sqrt3_d = sqrt(3) * d / params[i]
                K = K + (params[i + noVars] .* 1. .+ sqrt3_d) .* exp.(-sqrt3_d)
            elseif model[:kernelType] == "matern5/2"
                row, col = size(ND)
                d = sqrt.(abs.(ND))
                sqrt5_d = sqrt(5)*ones(row,col) + d / params[i]
                term2 = (5.0 * ND) / (3.0 * params[i]^2)
                K = K + params[i + noVars] .* (1. .+ sqrt5_d .+ term2) .* exp.(-sqrt5_d)
            elseif model[:kernelType] == "cauchy"
                row, col = size(ND)
                K = K + params[i + noVars] .* 1. ./ (ones(row, col) + (ND ./ params[i]^2))
            else 
                println("The kernel method is not available for individual calculations.")
            end
        end
    
    elseif model[:kernelVersion] == "family"
        ND  = normDiff(X, X)
        row, col = size(ND)
        d   = sqrt.(abs.(ND))
        K   = params[2] * exp.(-ND/(2*params[1]^2))
        K   = K .+ params[3] * exp.(-d/params[4])
        sqrt3_d = sqrt(3) * d / params[6]
        K   = K .+ params[5] * ((1. .+ sqrt3_d) .* exp.(-sqrt3_d))
        sqrt5_d = sqrt(5)*ones(row,col) + d / params[8]
        term2 = (5.0 * ND) / (3.0 * params[8]^2)
        K   = K .+ params[7] * ((1. .+ sqrt5_d .+ term2) .* exp.(-sqrt5_d))
        K   = K .+ params[10] * (1. ./ (ones(row, col) + (ND ./ params[9]^2)))

    else
        ND = normDiff(X, X)
        #println("Max asymmetry in ND: ", maximum(abs.(ND - ND')))
        if model[:kernelType] == "gaussian"
            K = exp.(-ND/(2*params[1]^2))
        elseif model[:kernelType] == "matern1/2"
            row, col = size(ND)
            d = sqrt.(abs.(ND))
            K = exp.(-d/params[1])
        elseif model[:kernelType] == "matern3/2"
            row, col = size(ND)
            d = sqrt.(abs.(ND))
            sqrt3_d = sqrt(3) * d / params[1]
            K = (1. .+ sqrt3_d) .* exp.(-sqrt3_d)
        elseif model[:kernelType] == "matern5/2"
            row, col = size(ND)
            d = sqrt.(abs.(ND))
            sqrt5_d = sqrt(5)*ones(row,col) + d / params[1]
            term2 = (5.0 * ND) / (3.0 * params[1]^2)
            K = (1. .+ sqrt5_d .+ term2) .* exp.(-sqrt5_d)
        elseif model[:kernelType] == "cauchy"
            row, col = size(ND)
            K = 1. ./ (ones(row, col) + (ND ./ params[1]^2))
        else
            println("Error! Model kerneltype incorrecty")
        end

    end
    
    if state == "training"
        nl   = size(K, 1)
        oneN = ones(nl, nl) / nl
        #if size(model[:initialParams]) == size(zeros(1))
        #println("Max asymmetry in K before centering: ", maximum(abs.(K - K')))
        K    = K .- oneN * K .- K * oneN .+ oneN * K * oneN 
        #println("Max asymmetry in K after centering: ", maximum(abs.(K - K')))
        K += 1e-10 * I
    else 
        nl   = size(K, 1)
        K   =  K 
    end

    return K
end

function kernelRBFTest(X::Matrix{Float64}, XTest::Matrix{Float64}, params, model::Dict{Symbol,Any})

    KCal = kernelRBF2(X, params, model, "test")

    if model[:kernelVersion] == "individual"
        noRows, noVars = size(X)
        noRowsT, noVarsT = size(XTest)

        K = zeros(noRowsT, noRows)
        
        for i = 1:noVars
            ND = normDiff(XTest[:,i], X[:,i])
            if model[:kernelType] == "gaussian"
                K = K + exp.(-ND)/(2*params[i]^2)
            elseif model[:kernelType] == "matern1/2"
                row, col = size(ND)
                d = sqrt.(abs.(ND))
                K = K + exp.(-d/params[i])
            elseif model[:kernelType] == "matern3/2"
                row, col = size(ND)
                d = sqrt.(abs.(ND))
                sqrt3_d = sqrt(3) * d / params[i]
                K = K + (1. .+ sqrt3_d) .* exp.(-sqrt3_d)
            elseif model[:kernelType] == "matern5/2"
                row, col = size(ND)
                d = sqrt.(abs.(ND))
                sqrt5_d = sqrt(5)*ones(row,col) + d / params[i]
                term2 = (5.0 * ND) / (3.0 * params[i]^2)
                K = K + (1. .+ sqrt5_d .+ term2) .* exp.(-sqrt5_d)
            elseif model[:kernelType] == "cauchy"
                row, col = size(ND)
                K = K + 1. ./ (ones(row, col) + (ND ./ params[i]^2))
            else 
                println("The kernel method is not available for individual calculations.")
            end
        end

    elseif model[:kernelVersion] == "individualScale"
        noRows, noVars = size(X)
        noRowsT, noVarsT = size(XTest)

        K = zeros(noRowsT, noRows)
        
        for i = 1:noVars
            ND = normDiff(XTest[:,i], X[:,i])
            if model[:kernelType] == "gaussian"
                K = K + params[i + noVars] .* exp.(-ND)/(2*params[i]^2)
            elseif model[:kernelType] == "matern1/2"
                row, col = size(ND)
                d = sqrt.(abs.(ND))
                K = K + params[i + noVars] .* exp.(-d/params[i])
            elseif model[:kernelType] == "matern3/2"
                row, col = size(ND)
                d = sqrt.(abs.(ND))
                sqrt3_d = sqrt(3) * d / params[i]
                K = K + params[i + noVars] .* (1. .+ sqrt3_d) .* exp.(-sqrt3_d)
            elseif model[:kernelType] == "matern5/2"
                row, col = size(ND)
                d = sqrt.(abs.(ND))
                sqrt5_d = sqrt(5)*ones(row,col) + d / params[i]
                term2 = (5.0 * ND) / (3.0 * params[i]^2)
                K = K + params[i + noVars] .* (1. .+ sqrt5_d .+ term2) .* exp.(-sqrt5_d)
            elseif model[:kernelType] == "cauchy"
                row, col = size(ND)
                K = K + params[i + noVars] .* 1. ./ (ones(row, col) + (ND ./ params[i]^2))
            else 
                println("The kernel method is not available for individual calculations.")
            end
        end
    
    elseif model[:kernelVersion] == "family"
        ND  = normDiff(XTest, X)
        row, col = size(ND)
        d   = sqrt.(abs.(ND))
        K   = params[2] * exp.(-ND/(2*params[1]^2))
        K   = K .+ params[3] * exp.(-d/params[4])
        sqrt3_d = sqrt(3) * d / params[6]
        K   = K .+ params[5] * ((1. .+ sqrt3_d) .* exp.(-sqrt3_d))
        sqrt5_d = sqrt(5)*ones(row,col) + d / params[8]
        term2 = (5.0 * ND) / (3.0 * params[8]^2)
        K   = K .+ params[7] * ((1. .+ sqrt5_d .+ term2) .* exp.(-sqrt5_d))
        K   = K .+ 1. ./ (ones(row, col) + (ND ./ params[9]^2))

    else
        ND = normDiff(XTest, X)
        if model[:kernelType] == "gaussian"
            K = exp.(-ND/(2*params[1]^2))
        elseif model[:kernelType] == "matern1/2"
            row, col = size(ND)
            d = sqrt.(abs.(ND))
            K = exp.(-d/params[1])
        elseif model[:kernelType] == "matern3/2"
            row, col = size(ND)
            d = sqrt.(abs.(ND))
            sqrt3_d = sqrt(3) * d / params[1]
            K = (1. .+ sqrt3_d) .* exp.(-sqrt3_d)
        elseif model[:kernelType] == "matern5/2"
            row, col = size(ND)
            d = sqrt.(abs.(ND))
            sqrt5_d = sqrt(5)*ones(row,col) + d / params[1]
            term2 = (5.0 * ND) / (3.0 * params[1]^2)
            K = (1. .+ sqrt5_d .+ term2) .* exp.(-sqrt5_d)
        elseif model[:kernelType] == "cauchy"
            row, col = size(ND)
            K = 1. ./ (ones(row, col) + (ND ./ params[1]^2))
        else
            println("Error! Model kerneltype incorrecty")
        end

    end

    n       = size(KCal, 1)
    oneN    = ones(n, n)/n
    nTest   = size(XTest, 1)
    oneNTest = ones(nTest, n)/n
    KTest       = K .- oneNTest * KCal .- K * oneN .+ oneNTest * KCal * oneN 
    return KTest
end

function sample(n::Int, sp::Float64)
    ns    = ceil(Int, n*sp)
    perm  = Random.shuffle(1:n)
    idxS = perm[1:ns]
    return idxS
end

function centerKernel(K::Matrix{Float64})
    nl   = size(K, 1)
    oneN = ones(nl, nl) / nl    
    cK   = K .- oneN * K .- K * oneN .+ oneN * K * oneN
    return cK
end

function compute_pca(x::Matrix{Float64}, dim::Int)
    m, n = size(x)
    # Check for NaN or infinite values
    if any(isnan, x) || any(isinf, x)
        error("Matrix contains NaN or infinite values")
    end

    if m < n
        p_adj, s, u = svd(x')
    else
        u, s, p_adj = svd(x)
    end
    t = u * Diagonal(s)
    eigenvalues = s .^ 2 
    return t[:,1:dim], Matrix(p_adj)[:,1:dim], eigenvalues[1:dim]
end

function compute_pcr(K::Matrix{Float64}, Y::Vector{Float64}, dim::Int)
    ε = 1e-8  # Small regularization constant
    K_reg = K + ε * I
    T, P = compute_pca(K, dim)               # 1. Compute KPCA
    Bsmall = T\Y       # 2. Weights in KPC space
    B = P * Bsmall     # 3. Weights in K space
    return B
end

function computeEigs(cK::Matrix{Float64})
   # println("Checking matrix before eigenvalue decomposition...")

    # Check for NaNs and Infs
    if any(isnan.(cK))
        error("NaN detected in cK! Something is wrong with the kernel matrix.")
    end
    if any(isinf.(cK))
        error("Inf detected in cK! Kernel parameters might be too extreme.")
    end  # <--- Missing `end` was here

    # Check if the matrix is symmetric (should be for covariance/kernel matrices)
   # if !issymmetric(cK)
    #    println("Warning: cK is not symmetric! Making it symmetric...")
    #    cK = (cK + cK') / 2  # Enforce symmetry
    #end

    # Compute eigenvalues and eigenvectors
    eig = eigen(real(cK))
    λ = eig.values
    P = eig.vectors
    idx = sortperm(λ, rev=true)
    return P[:, idx], λ[idx]
end

function rho_average(params::Vector{Float64}, XHealthy::Matrix{Float64}, XFaulty::Matrix{Float64}, model::Dict{Symbol,Any}, T²lim::Float64, P::Matrix{Float64}, λ::Vector{Float64})
    # Compute `ρ` in a single expression using list comprehension
    if model[:paper] == 1

        ρ_vals = [
            let
            XHSam   = XHealthy[Int.(model[:idxSamp][:,i]), :]
            XFSam   = XFaulty[Int.(model[:idxSamp][:,i]), :]
            XMixed  = vcat(XHSam, XFSam)
            nH      = size(XHSam, 1)  
            nF      = size(XFSam, 1)  
            yi      = vcat(zeros(nH), ones(nF)) 
            # calibration
            K       = kernelRBF2(XMixed, params, model, "training")
            B       = compute_pcr(K, yi, model[:dim])
            # test 
            Kₛ      = kernelRBFTest(XMixed, model[:XTest], params, model)   
            ypred   = Kₛ*B
            computeLoss2(model[:YTest], ypred)
        end
            for i in 1:model[:nsamp]
        ]
    end

    return sum(ρ_vals) / model[:nsamp]  # Ensure scalar output
end


function grad_rho(model::Dict{Symbol, Any})
    params  = model[:initialParams]
    beta    = 0.9
    beta2   = 0.5
    params  = Float64.(model[:initialParams])
    threshold = 1
    model[:np] = size(params,1)

    parameterHistory= zeros(Float64, model[:iter], model[:np])
    gradHistory     = zeros(Float64, model[:iter], model[:np])
    lossHistory     = zeros(Float64, model[:iter], 1)

    for i = 1:model[:iter]
    rowData = size(model[:XFaulty], 1)
    model[:idxBatch] = sort(sample(rowData, 1.0))
    model[:idxSamp] = zeros(ceil(Int, length(model[:idxBatch])*model[:sp]), 0)

    KBatch = kernelRBF2(model[:XHealthy][model[:idxBatch],:], exp.(params), model, "training")

    T, P, λ = compute_pca(KBatch, model[:dim])
    #P, λ   = computeEigs(KBatch)
    #T      = computeScores(KBatch, P, model[:dim])

    T²lim  = computeT2Limit(T, λ, model[:dim], model[:α])

    #println(i)
    #println("exp(params): ", exp.(params))
    if mode(i%10) == 0
        println(i)
    end

    for i = 1: model[:nsamp]
        model[:idxSamp] = hcat(model[:idxSamp], sort(sample(length(model[:idxBatch]), model[:sp])))
    end

    for j = 1 : model[:np] 
        parameterHistory[i,j] = params[j]
        #println(params[j])
    end

    ρ, ∇ = withgradient(param -> rho_average(exp.(param), model[:XHealthy][model[:idxBatch],:], model[:XFaulty][model[:idxBatch],:], model, T²lim, P, λ), params)

    ∇ = model[:gradClip] .* ∇./norm(∇)
    lossHistory[i] = ρ
    
        
    for j = 1:model[:np]
        if i > 1
            gradHistory[i,j]      = ∇[1][j] 
        else
            gradHistory[i,j]      = 0.0
        end
        if i == 1
            params[j] = params[j] - model[:learnRate] * ∇[1][j] 
        else
            params[j] = params[j] - model[:learnRate] * (∇[1][j] + beta * (params[j] - parameterHistory[i-1,j]))
        end
    end
    
    end

    return model, parameterHistory, lossHistory, gradHistory
end

function optimize_parameters(model::Dict{Symbol, Any})
    
    model, parameterHistory, lossHistory, gradHistory = grad_rho(model)

    model[:runningLoss] = movmean(lossHistory, 10)
    _, model[:bestLoss] = findmin(model[:runningLoss])
    model[:bestParam] = zeros(model[:np], 1)

    for i = 1:model[:np]
        model[:bestParam][i] = parameterHistory[model[:bestLoss], i]
    end
    return model, parameterHistory, lossHistory, gradHistory
end

function computeScores(cK::Matrix{Float64}, P::Matrix{Float64}, noPCs::Int)
    T = cK * P[:, 1:noPCs]
    return T
end

function computeT2(T::Matrix{Float64}, λ, noPCs)
    #println("Size of T: ", size(T))  # Debugging output
    #println("Size of λ: ", size(λ))  # Debugging output
    #println("Values of λ: ", λ[1:noPCs])  # Check eigenvalues
    #println("Values of 1.0 ./ λ: ", 1.0 ./ λ[1:noPCs])  # Check inverse eigenvalues

    inv_λ = 1.0 ./ λ[1:noPCs]  # Compute inverse eigenvalues
    #println("Size of inv_λ: ", size(inv_λ))  # Debugging output

    T² = sum((T .* inv_λ') .* T, dims=2)  # Compute T² for each sample
    #println("Values of T² before vec: ", T²)  # Debugging output
    return vec(T²)  # Ensure it's a 1D vector
end

function computeSPEx(cK::Matrix{Float64}, P::Matrix{Float64}, T::Matrix{Float64})
    SPEₓ = sum((cK * P).^2, dims=2) .- sum(T.^2, dims=2)
    return SPEₓ
end

function computeSPExLimit(cK::Matrix{Float64}, P::Matrix{Float64}, T::Matrix{Float64}, α)
    SPEₓ = computeSPEx(cK, P, T)
    a = mean(SPE)
    b = var(SPE)
    g = b / (2 * a)
    h = 2 * a^2 / b
    chi_squared_dist = ChiSq(h)
    SPEₓlim = g * quantile(chi_squared_dist, α)
    return SPEₓlim
end

function computeT2Limit(T::Matrix{Float64}, λ, noPCs::Int, α)
    T² = computeT2(T, λ, noPCs)  # T² is now a 1D vector

   if model[:α] == 0.95
        T²lim =  mean(T²) + 2*std(T²)
   elseif model[:α] == 0.99
        T²lim =  mean(T²) + 3*std(T²)
   end

    return T²lim
end


function smooth_indicator(x, y; sharpness=1000)
    return 1.0 .- (1.0 ./ (1.0 .+ exp.(sharpness .* (x .- y))))
end

function computeLoss(yi, T²test, T²lim)
    # Ensure yi is a vector
    yi = vec(yi)

    # Get indices for faulty and healthy samples
    fault_idx = findall(yi .== 1)
    healthy_idx = findall(yi .== 0)

    #println("Faulty count: ", length(fault_idx))
    #println("Healthy count: ", length(healthy_idx))

    # Ensure non-empty subsets
    if isempty(fault_idx) || isempty(healthy_idx)
        println("Warning: Empty subset in computeLoss! Returning neutral loss value.")
        return 0.5  # Neutral value to keep optimization going
    end

    T2_fault   = T²test[fault_idx]
    T2_healthy = T²test[healthy_idx]

    ηₙ = mean(smooth_indicator(T2_fault, T²lim))   # Proportion of faulty points ABOVE limit
    ηᵤ = mean(1.0 .- (smooth_indicator(T2_healthy, T²lim)))  # Proportion of healthy points BELOW limit (flipped)

    #println("ηₙ (fault detection rate): ", ηₙ)
    #println("ηᵤ (healthy misclassification rate): ", ηᵤ)
    #println("Final loss: ", 1.0 - (ηₙ + ηᵤ) / 2.0)

    ρ = 1.0 - (ηₙ + ηᵤ) / 2.0
    return ρ
end

function computeLoss2(yi, ypred; sharpness=1000.0)
    # Ensure yi is a vector
    yi = vec(yi)

    # Get indices for faulty and healthy samples
    fault_idx = findall(yi .== 1)
    healthy_idx = findall(yi .== 0)

    # Ensure non-empty subsets
    if isempty(fault_idx) || isempty(healthy_idx)
        println("Warning: Empty subset in computeLoss2! Returning neutral loss value.")
        return 0.5  # Neutral value to keep optimization going
    end

    ypred_faulty = ypred[fault_idx]
    ypred_healthy = ypred[healthy_idx]

    # Smooth indicators for classification (assuming threshold at 0.5 for classification)
    correct_healthy = mean(1.0 .- smooth_indicator(ypred_healthy, 0.5; sharpness=sharpness)) # 1 - P(faulty)
    correct_faulty = mean(smooth_indicator(ypred_faulty, 0.5; sharpness=sharpness)) # P(faulty)

    # Proportion of correctly classified healthy samples
    ρ = 1 - (correct_healthy + correct_faulty) / 2.0

    return ρ
end


function movmean(x, m) 
    # this comes from github.com/chmendoza
    """Compute off-line moving average
    
    Parameters
    ----------
    x (Array):
        One-dimensional array with time-series. size(x)=(N,)
    m (int):
        Length of the window where the mean is computed
    
    Returns
    -------
    y (Array):
        Array with moving mean values. size(y)=(N-m+1)    
    """

    N = length(x)
    y = fill(0., N-m+1)

    y[1] = sum(x[1:m])/m
    for i in 2:N-m+1
        @views y[i] = y[i-1] + (x[i+m-1]-x[i-1])/m
    end
    return y
end