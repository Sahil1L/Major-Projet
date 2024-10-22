% Load file and handle missing fields in HV data (STEP 1)
file_name = 'subset_27.mat';  % Change this filename when using a different file
loaded_data = load(file_name);

% Check what variables are loaded
var_names = fieldnames(loaded_data);

% Dynamically assign dataset variable based on loaded data
if ~isempty(var_names)
    dataset = loaded_data.(var_names{1});  % Assign the first variable found in the loaded file to dataset
else
    error('No variables found in the loaded file.');
end

% Initialize counters for missing data
missing_HV = [];
missing_MV = [];
incompatible_sizes = [];

% Initialize data storage for plotting later
HV_x = [];
MV_x = [];
HV_vx = [];
MV_vx = [];
actions_HV = [];
rewards = [];

% Determine the number of entries in the dataset
num_entries = length(dataset);

for i = 1:num_entries
    % Check if the field 'HV' exists in the structure
    if isfield(dataset(i), 'HV')
        HV_data = dataset(i).HV;
        
        % Check if the necessary fields exist in HV_data
        required_HV_fields = {'x', 'y', 'vx', 'vy', 'ax', 'ay'};
        if all(isfield(HV_data, required_HV_fields))
            HV_x = [HV_x; HV_data.x];  % Collect HV position data for later plotting
            HV_vx = [HV_vx; HV_data.vx];  % Collect HV velocity data for later plotting
        else
            disp(['Missing fields in HV data for entry ', num2str(i)]);
            missing_HV = [missing_HV, i];
            continue; % Skip this entry if any fields are missing
        end
    else
        disp(['HV field is not available in the structure for entry ', num2str(i)]);
        missing_HV = [missing_HV, i];
        continue; % Skip to the next iteration if HV data is not available
    end

    % Extract data for the current entry
    if isfield(dataset(i), 'MV')
        MV = dataset(i).MV;
        
        % Check if the fields exist before accessing them
        required_MV_fields = {'x', 'y', 'vx', 'vy', 'ax', 'ay'};
        if all(isfield(MV, required_MV_fields))
            MV_x = [MV_x; MV.x];  % Collect MV position data for later plotting
            MV_vx = [MV_vx; MV.vx];  % Collect MV velocity data for later plotting
        else
            disp(['Fields missing in MV entry ', num2str(i)]);
            missing_MV = [missing_MV, i];
            continue; % Skip this iteration if fields are missing
        end
    else
        disp(['MV field is not available in the structure for entry ', num2str(i)]);
        missing_MV = [missing_MV, i];
        continue; % Skip to the next iteration if MV data is not available
    end

    % Check for array size compatibility
    if length(HV_x) ~= length(MV_x) || length(HV_vx) ~= length(MV_vx)
        disp(['Incompatible sizes in entry ', num2str(i)]);
        incompatible_sizes = [incompatible_sizes, i];
        continue; % Skip this iteration if array sizes are incompatible
    end
    
    %% Define Discretization Parameters (STEP 2)
    s1_min = -150;  % Minimum relative position (meters)
    s1_max = 150;   % Maximum relative position (meters)
    s2_min = -20;   % Minimum relative velocity (m/s)
    s2_max = 20;    % Maximum relative velocity (m/s)

    delta_s1 = 2;   % Step size for relative position
    delta_s2 = 2;   % Step size for relative velocity

    s1_values = s1_min:delta_s1:s1_max;
    s2_values = s2_min:delta_s2:s2_max;

    numStates = length(s1_values) * length(s2_values);

    % Function to map (s1, s2) to a unique state index
    state_to_index = @(s1, s2) (find(s1_values == s1)) + (find(s2_values == s2) - 1) * length(s1_values);

    % Define other parameters
    K = 60; % Strategic planning horizon
    deltaT = 0.15; % Time step (seconds)
    time_steps = linspace(0, deltaT*K, K);  % New time vector for 30 steps

    actions_HV = [-1, 0, 1]; % HV actions: decelerate, maintain speed, accelerate

    % Acceleration threshold for comfort (e.g., 0.5 m/s^2)
    acceleration_threshold = 0.5;

    % Define the comfort penalty function
    comfort_penalty = @(a) ((abs(a) > acceleration_threshold) * ((abs(a) - acceleration_threshold)^2)*10);

    % Define Reward Function with comfort and safety
    rel_pos_threshold = 5;   % Example safety threshold for relative position (meters)
    rel_vel_threshold = 5;   % Example safety threshold for relative velocity (m/s)
    
    reward_function = @(s, aH)  -((s(1) + aH)^2 + (s(2) + aH)^2) ...  % Penalize large deviations in position/velocity
                                - ((s(1) < rel_pos_threshold) * 10) ...  % High penalty if relative position is below safe threshold
                               
                             

    % Define Transition Function for HV only (MV is predefined)
    transition_function = @(s, aH) [s(1) + s(2)*deltaT - 0.5*aH*deltaT^2, ...
                                    s(2) - aH*deltaT];  % HV dynamics (s1: position, s2: velocity)

    % Initialize Value Function and Q-values for HV only
    V = zeros(numStates, K+1); % Initialize value function
    Q_H = zeros(numStates, length(actions_HV), K); % Q-values for HV (Follower)

    % A. Strategic Planner Implementation (STEP 4)
    %  Backward recursion for value function calculation
    for k = K:-1:1
        for s = 1:numStates
            [s1, s2] = ind2sub([length(s1_values), length(s2_values)], s); % Convert index to state variables
            current_s1 = s1_values(s1);
            current_s2 = s2_values(s2);
            
            for aH = 1:length(actions_HV)
                % Compute next state
                next_s = transition_function([current_s1, current_s2], actions_HV(aH));
                
                % Boundary conditions: Clamp next state within defined ranges
                next_s(1) = min(max(next_s(1), s1_min), s1_max);
                next_s(2) = min(max(next_s(2), s2_min), s2_max);
                
                % Find the closest discrete state
                [~, idx_s1] = min(abs(s1_values - next_s(1)));
                [~, idx_s2] = min(abs(s2_values - next_s(2)));
                next_state = (idx_s2 - 1) * length(s1_values) + idx_s1;
                
                % Compute Q-value for HV
                Q_H(s, aH, k) = reward_function([current_s1, current_s2], actions_HV(aH)) + V(next_state, k+1);
            end
            
            % Optimal action for HV
            [V(s, k), ~] = max(Q_H(s, :, k));
        end
    end


    % B. Tactical Planner Implementation (Optimizing HV Only)
    M = 20; % Example horizon for tactical planning

    V_tactical = zeros(numStates, M+1); % Initialize tactical value function
    Q_tactical = zeros(numStates, length(actions_HV), M); % Initialize Q-values for HV only (no MV optimization)

    % Backward recursion for tactical value function
    for t = M:-1:1
       for s = 1:numStates
              [s1, s2] = ind2sub([length(s1_values), length(s2_values)], s); % Convert index to state variables
              current_s1 = s1_values(s1);
              current_s2 = s2_values(s2);
        
        % Use MV data directly from the dataset for this entry
        MV_x_t = MV_x(i);   % Get MV position at time t
        MV_vx_t = MV_vx(i); % Get MV velocity at time t
        
        % Loop over HV actions only (actions_HV)
        for aH = 1:length(actions_HV)
            % Compute next state for HV using the current MV data
            next_s = transition_function([current_s1, current_s2], actions_HV(aH));
            
            % Boundary conditions: Clamp next state within defined ranges
            next_s(1) = min(max(next_s(1), s1_min), s1_max);
            next_s(2) = min(max(next_s(2), s2_min), s2_max);
            
            % Find the closest discrete state
            [~, idx_s1] = min(abs(s1_values - next_s(1)));
            [~, idx_s2] = min(abs(s2_values - next_s(2)));
            next_state = (idx_s2 - 1) * length(s1_values) + idx_s1;
            
            % Compute Q-value for HV (using MV data from the dataset)
            Q_tactical(s, aH, t) = reward_function([current_s1, current_s2], actions_HV(aH)) ...
                                   + V_tactical(next_state, t+1);
        end
        
        % Choose the best action based on tactical Q-values
        [max_Q_tactical, ~] = max(Q_tactical(s, :, t), [], 'all');
        V_tactical(s, t) = max_Q_tactical;
    end
  end

end

% Initialize policy table
policy_table = zeros(numStates, K);  % Rows correspond to states, columns to time steps

for k = 1:K
    for s = 1:numStates
        % Find the action with the maximum Q-value for this state and time step
        [~, optimal_action_idx] = max(Q_H(s, :, k));
        policy_table(s, k) = actions_HV(optimal_action_idx);  % Store the optimal action
    end
end

% Save policy table
save('policy_table.mat', 'policy_table'); 

s1_min = -150; s1_max = 150; delta_s1 = 2;
s2_min = -20;  s2_max = 20;  delta_s2 = 2;

% Define state space for s1 (relative distance) and s2 (relative velocity)
s1_values = s1_min:delta_s1:s1_max;
s2_values = s2_min:delta_s2:s2_max; 

% Create a grid of all possible states
[state_s1, state_s2] = meshgrid(s1_values, s2_values);

% Combine the state parameters into one array (for simplicity, it's reshaped)
state_matrix = [state_s1(:), state_s2(:)];

% Now each state has a corresponding action in the policy_table (row-wise mapping)
% policy_table should be mapped to state_matrix
policy_table_with_states = [state_matrix, policy_table];  % Combine states with the actions

save('policy_table_with_states.mat', 'policy_table_with_states');

% Summary of missing/incompatible entries
disp(['Total entries with missing HV data: ', num2str(length(missing_HV))]);
disp(['Total entries with missing MV data: ', num2str(length(missing_MV))]);
disp(['Total entries with incompatible sizes: ', num2str(length(incompatible_sizes))]);


