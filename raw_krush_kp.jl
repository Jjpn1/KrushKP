begin
		using CSV
		using DataFrames
		using HTTP
		using MolecularGraph
		using MoleculeFlow
		using Graphs
		using StatsBase
		using Flux
		using LaplaceRedux
		
end

# ╔═╡ 6b92fb9a-28dc-11f1-b822-b309dc33f338
print("This notebook is to fully demonstrate the Krush_KP AI pipeline")

# ╔═╡ 78f3cbe8-0fc3-4537-afd6-a4c140894892
begin
	# Loading the dataset
	url = "https://raw.githubusercontent.com/Jjpn1/KrushKP/refs/heads/main/sample_data_krush_KP.csv"
	response = HTTP.get(url)
	df = CSV.read(response.body, DataFrame)
	
	# Extracting the target variable (5th column)
	# In Julia, indices start at 1, so column index 5 is used
	y = df[:, 4]
	
	# Display the first few rows
	first(df, 4)
end

# ╔═╡ 0b2c76f0-ace7-4267-94b7-fb1e252f9486
# ╠═╡ disabled = true
#=╠═╡
function kalculate_properties(smiles)
	mol = mol_from_smiles(smiles)
	if mol == Nothing
		return Nothing
	end

	g = mol_to_graph(mol)
	number_of_bonds = nv(g)
  ╠═╡ =#

# ╔═╡ 798e951e-5585-430f-9638-0df4911a9b79
calculate_properties(::Missing) = missing

# ╔═╡ f5e48d41-f530-4e2d-ac1f-1d08d347dd28
function calculate_properties(smiles::String)
    mol = try
        smilestomol(smiles)
    catch
        # Let's also return `missing` here instead of `nothing` so the DataFrame 
        # stays consistent if a SMILES string is invalid!
        return missing 
    end
    
    # Use ne(mol) instead of edgecount(mol)
    # Alternatively, you can use `bondcount(mol)` or `length(collect(edges(mol)))`
    num_bonds = ne(mol) 
    
    if num_bonds == 0
        return 0.0
    end
    
    symbols = atom_symbol(mol)
    
    # edges(mol) returns an iterator of the bonds
    num_cc_pairs = count(edges(mol)) do e
        # src(e) and dst(e) get the atom indices at either end of the bond
        string(symbols[src(e)]) == "C" && string(symbols[dst(e)]) == "C"
    end
    
    return num_cc_pairs / num_bonds
end

# ╔═╡ bcadaf53-ffc3-4545-baeb-579c5eeb5539
#Test for calculate properties function    
calculate_properties("CCO")

# ╔═╡ c646625f-6445-4592-a8f3-618228596830
begin 
df.frac_CC = calculate_properties.(df.smiles_sequence)

# And your scaling step remains exactly the same:
med = median(skipmissing(df.frac_CC))
iqr_value = iqr(skipmissing(df.frac_CC))
df.frac_CC = (df.frac_CC .- med) ./ iqr_value
end

# ╔═╡ d02b2ab5-30ee-4e29-b74e-1956e4406d66
begin
mol = smilestomol("CCO")

# Assign directly to a single variable
mass = standard_weight(mol)

# The same fix applies if you are using exactmass:
exact_m = exactmass(mol)

println("Standard Weight: ", mass)
println("Exact Mass: ", exact_m)
end

# ╔═╡ a4125adf-e30f-4b5d-a9f4-3b5245c76144
begin
    function generate_fingerprints(smiles::String)
        # 1. Reverted back to MoleculeFlow's native parser
        mol = try
            mol_from_smiles(smiles)
        catch
            return missing
        end
        
        # 2. Safety check
        if mol === nothing || ismissing(mol)
            return missing
        end
        
        # 3. Generate the fingerprint (This will now work!)
        fp = morgan_fingerprint(mol)
        
        return Float32.(fp)
    end
    
    # Fallback for missing data
    generate_fingerprints(::Missing) = missing
end

# ╔═╡ 6b7a5035-6440-4b15-abe1-5277698d886f
begin
    # Safely assign the generated vector of vectors into a new column
    df[!, :fingerprints] = generate_fingerprints.(df.smiles_sequence)
end

# ╔═╡ d7f8212d-d8a5-4927-9bf1-4bc2868ea8dd
df.fingerprints

# ╔═╡ 12b862e5-6936-4229-a035-3ca2ef74bc9b
first(df, 4)

# ╔═╡ 998a4fd9-acd2-4895-902d-e0c548c69031
"""This is for the machine learning part I only use MLP here and Bayesian Neural networks with Laplace redux"""

# ╔═╡ 03b60bf1-9893-49ac-a7d2-1e2d572610dc
begin
    # 1. Drop rows with invalid/missing fingerprints
    clean_df = dropmissing(df, :fingerprints)
    
    # 2. Extract Features (X)
    X_matrix = reduce(hcat, clean_df.fingerprints)
    
    # 3. Extract Targets (y)
    # Using the exact column name you specified
    y_raw = clean_df.result
    
    # Convert the 0s and 1s directly to Float32 and reshape into a (1 x N) Matrix
    y_matrix = reshape(Float32.(y_raw), 1, length(y_raw))
    
    # 4. Create the DataLoader
    train_loader = Flux.DataLoader((X_matrix, y_matrix), batchsize=32, shuffle=true)
    
    println("X shape: ", size(X_matrix))
    println("y shape: ", size(y_matrix))
end

# ╔═╡ a1277e86-cb2d-4f53-a694-4c900089692e
begin
    # Build the neural network
    mlp_model = Chain(
        # Input layer expects the 2048-bit Morgan fingerprint
        Dense(2048 => 512, relu),
        Dropout(0.3),
        
        # Hidden layer
        Dense(512 => 128, relu),
        Dropout(0.3),
        
        # Output layer for binary classification
        Dense(128 => 1, sigmoid)
    )
end


# ╔═╡ d11dc101-a1f6-4ac7-ab2f-87f4affd1fc4
begin
    # Define the loss function
    loss_fn(m, x, y) = Flux.Losses.binarycrossentropy(m(x), y)
    
    # Setup the optimizer state (Learning rate of 0.001)
    opt_state = Flux.setup(Adam(1e-3), mlp_model)
    
    epochs = 20
    epoch_losses = Float64[]
    
    # Training Loop
    for epoch in 1:epochs
        batch_loss = 0.0
        
        for (x_batch, y_batch) in train_loader
            # Calculate gradients
            val, grads = Flux.withgradient(mlp_model) do m
                loss_fn(m, x_batch, y_batch)
            end
            
            # Update the weights
            Flux.update!(opt_state, mlp_model, grads[1])
            batch_loss += val
        end
        
        # Store average loss for this epoch
        avg_loss = batch_loss / length(train_loader)
        push!(epoch_losses, avg_loss)
    end
    
    # Output the final loss array to visualize the training curve
    epoch_losses
end

# ╔═╡ 38c9b691-2c63-4ce0-9353-49c536741912
begin
    function predict_antimicrobial(smiles_string::String)
        # 1. Generate the fingerprint using our existing safe function
        fp = generate_fingerprints(smiles_string)
        
        # 2. Safety check: Did the SMILES string fail to parse?
        if ismissing(fp)
            return "Error: Invalid SMILES string. Could not generate fingerprint."
        end
        
        # 3. Reshape the 1D vector into the 2D matrix required by Flux (2048 x 1)
        x_input = reshape(fp, 2048, 1)
        
        # 4. Pass it through the trained network
        # The model outputs a 1x1 matrix, so we grab the first element [1]
        probability = mlp_model(x_input)[1]
        
        # 5. Interpret the threshold (0.5 is standard for binary classification)
        is_active = probability >= 0.5
        
        # 6. Format the output nicely
        status = is_active ? "Active against K. pneumoniae" : "Inactive"
        
        return (
            SMILES = smiles_string, 
            Prediction = status, 
            Probability = round(probability, digits=4)
        )
    end
    
    # Let's test it immediately! (Using Aspirin as a random example)
    predict_antimicrobial("CC(C)C[C@H](NC(=O)[C@@H](NC(=O)[C@H](C)NC(=O)CNC(=O)[C@@H](NC(=O)[C@H](CCCC[NH3+])NC(=O)[C@H](CCCC[NH3+])NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CCCC[NH3+])NC(=O)[C@@H](NC(=O)[C@@H]([NH3+])CCCC[NH3+])Cc1c[NH]c2ccccc21)[C@@H](C)CC)C(C)C)C(=O)N[C@@H](CCCC[NH3+])C(=O)N[C@H](C(=O)N[C@@H](CC(C)C)C(N)=O)C(C)C")
end

# ╔═╡ 9748db53-5e76-405b-9a39-02db7c2652ec
println(mlp_model)

# ╔═╡ 8c373cd5-4e78-41c4-a0c5-8f080563f73d
begin
    # 1. Wrap the pre-trained Flux MLP model
    # We specify :classification because the target is binary (active vs inactive)
    la_model = Laplace(mlp_model; likelihood=:classification)
    
    # 2. Fit the Laplace approximation
    # This passes your training batches back through the network to calculate 
    # the Hessian matrix, defining the uncertainty/variance of the weights.
    LaplaceRedux.fit!(la_model, train_loader)
    
    # 3. Optimize the prior variance
    # This ensures the model's uncertainty calibration is mathematically sound 
    # and prevents the distributions from becoming overly broad.
    optimize_prior!(la_model)
    
    println("✅ Bayesian Laplace Approximation successfully fitted!")
end

# ╔═╡ 67f23496-088b-4759-9598-2849708c388f
begin
    function predict_bayesian_antimicrobial(smiles_string::String)
        # 1. Generate the 2048-bit Morgan fingerprint 
        fp = generate_fingerprints(smiles_string)
        
        if ismissing(fp)
            return "Error: Invalid SMILES string. Could not generate fingerprint."
        end
        
        # 2. Reshape the vector into the 2D matrix required by Flux
        x_input = reshape(fp, 2048, 1)
        
        # 3. Pass the fingerprint through the Laplace-wrapped model
        # This returns the marginalized predictive posterior probability
        bayesian_prob = LaplaceRedux.predict(la_model, x_input)[1]
        
        # 4. Interpret the output
        is_active = bayesian_prob >= 0.5
        status = is_active ? "Active against K. pneumoniae" : "Inactive"
        
        return (
            SMILES = smiles_string, 
            Prediction = status, 
            Bayesian_Probability = round(bayesian_prob, digits=4)
        )
    end
    
    # Test the Bayesian predictor with a sample string
    predict_bayesian_antimicrobial("CC(=O)Oc1ccccc1C(=O)O")
end
