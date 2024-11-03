#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef unsigned char u8;
typedef unsigned int u32;
typedef float f32;

typedef struct Item Item;
struct Item {
	f32 weight;
	f32 value;
};

typedef struct Simulated_Annealing_Parameters Simulated_Annealing_Parameters;
struct Simulated_Annealing_Parameters {
	u32 iteration_count;      // Total number of iterations for simulated annealing.
	u32 epoch_size;           // Number of iterations that will pass before the next temperature adjustment.
	f32 temperature;          // Starting temperature value.
	f32 temperature_scalar;   // Scalar "a" that adjusts temperature in every epoch (T(n+1) = a*T(n)).
	f32 minimal_temperature;  // Lower bound for temperature after which if won't be further adjusted.
	f32 cost_scalar;          // Used to adjust cost difference mapping after normalization.
	f32 boltzmann_scalar;     // Used to scale Boltzmann distribution.
};

// =========================================================================================================
// HELPER AND DEBUG FUNCTIONS
// =========================================================================================================

Item* generate_random_items(u32 n, f32 max_item_weight, f32 max_item_value) {
	Item* items = malloc(n * sizeof(Item));
	if(items) {
		for(u32 i = 0; i < n; ++i) {
			items[i].weight = ((f32)rand()/(f32)RAND_MAX * max_item_weight);
			items[i].value = ((f32)rand()/(f32)RAND_MAX * max_item_value);
		}
	}

	return items;
}

u8 random_u8() {
	return rand() & 1;
}

f32 random_f32() {
	return (f32)rand()/(f32)RAND_MAX;
}

u32 random_int_in_range_exclusive(u32 min, u32 max) {
	return min + rand() % (max - min);
}

f32 boltzmann_distribution(f32 scalar, f32 cost_difference, f32 temperature) {
	return scalar*exp(-cost_difference/temperature);
}

void print_items(Item* items, u32 n) {
	for(u32 i = 0; i < n; ++i) {
		printf("( %9.6lf, %9.6lf )\n", items[i].weight, items[i].value);
	}
}

void print_buffer(Item* buffer, u32 n, u32 knapsack_space) {
	for(u32 i = 0; i < knapsack_space; ++i) {
		printf("%3d ", i);
	}
	printf("\n");
	for(u32 i = 0; i < knapsack_space; ++i) {
		printf("====");
	}
	for(u32 i = 0; i < n; ++i) {
		if(i % knapsack_space == 0) {
			printf("\n");
		}
		printf("%3d ", (u32)buffer[i].value);
	}
	printf("\n");
}

// =========================================================================================================
// MOST BASIC RECURSIVE SOLUTION
//
// Time complexity  : O(2^n)
// Space complexity : O(2^n * (stack frame size)) for total used memory
//                    O(n * (stack frame size))   for memory used at arbitrary time
// =========================================================================================================

// Returns the maximum value that can be achieved while not exceeding knapsack space.
// Also returns maximum reached weight.

// This algorithm moves recursively from the last item in a greedy fashion ie. when
// deciding whether to include current item or not, it will always take the route
// that gives it maximum value at the current step (local maximum).

// This is not efficient solution, since it does redundant computations.
// It struggles even with small sample size that are not trivial (like n = 50, knapsack_space = 500).
// This is because in that case, if items can't reduce space to zero, algorithm will recurse
// all the way down until n = 0 (which is horrible for even a small n).

// This solution does not require additional explicit space for storing computation information,
// but that is of little relevancy when it eats stack with function stack frames.
// For any larger input value, default stack size can get exceeded.

// Asymptotic time complexity is O(2^n) since we traverse all possible paths and this can lead to
// repetitions, where we calculate function value multiple times for the same input.

Item recursive_solution(Item* items, u32 n, f32 knapsack_space) {
	// If no more space or no more items.
	if(knapsack_space <= 0 || n == 0) {
		return (Item){0, 0};
	}
	
	// Exclude currently last item if its weight exceeds knapsack space and recurse further.
	if(items[n-1].weight > knapsack_space) {
		return recursive_solution(items, n-1, knapsack_space);
	}

	// Go through cases where we exclude currently last item and where we include it.
	// If we include it, then we count its value and weight and reduce current knapsack space.
	
	// We pick the path that gives maximum value for the current step (local computational maximum).
	Item value_excluded = recursive_solution(items, n-1, knapsack_space);
	Item value_included = recursive_solution(items, n-1, knapsack_space - items[n-1].weight);
	value_included.weight += items[n-1].weight;
	value_included.value  += items[n-1].value;
	
	if(value_excluded.value > value_included.value) {
		return value_excluded;
	}
	else {
		return value_included;
	}
}

// =========================================================================================================
// RECURSIVE SOLUTION WITH 2D TABLE (ITEM WEIGHTS AND KNAPSACK WEIGHT MUST BE INTEGERS IN THIS APPROACH)
//
// Time complexity  : O(n*total_knapsack_space)
// Space complexity : O(n*total_knapsack_space)
// =========================================================================================================

// This algorithm is the same as the most basic recursive one, but it uses additional buffer to
// avoid doing computations that were done previously. This is done by recognizing that different
// recursive calls to the function only differ in two arguments, "n" and "knapsack_space". That
// means that if we form a grid of all possible value pairs (n, knapsack_space), we can store all
// possible computation results of all recursive calls that we encounter, thus saving computation time
// if the same recursive calls happen later.

// It is irrelevant (from asymptotic point of view) how we organize this grid.
// Here, grid is just an array that is equivalent to the matrix with the first coordinate being
// number of items remaining, and the second being remaining weight. Again, this interpretation is
// irrelevant and not needed since we just need buffer of dimension n*total_knapsack_space.

// In this form, it only works if weights are integer numbers, since they are used in buffer indexing.

// If weights are not integer numbers, then one possibility is to map them to integers, thus accepting
// small error but gaining speed.
// Another problem is that weights can be very large numbers or be scattered (have large variance).
// In that case, we would need a mapping that would move them to lower value in the case they are
// large but nicely grouped, or a mapping that would essentially be a non-cryptographic hash in the
// case of large variance.

// Compared to the most basic recursive algorithm that is exponential in time, this one is quadratic,
// which is a major difference.
// The price we pay for that is that we need space for the table that has quadratic complexity.
// Of course, in specific situations, we could also experiment with different table sizes in
// range [0, (n*total_knapsack_space)], thus taking more granular tradeoff between space and time
// complexity.
// Additionally, like the basic recursive solution, we need to take into account the default stack
// size if the input is large.

Item solve1(Item* items, u32 n, u32 knapsack_space, Item* buffer, u32 total_knapsack_space) {
	// If no more space or no more items.
	if(knapsack_space <= 0 || n == 0) {
	 	return (Item){0, 0};
	}

	// Just an index calculation since we represent buffer as an array instead of matrix.
	u32 current_item_index = ((n-1) * (total_knapsack_space + 1) + knapsack_space);

	// Buffer items values are initialized to -1, so if the value is not -1, then we have already
	// calculated that entry and we can just return it without again recursively calling function.
	if(buffer[current_item_index].value >= 0) {
		return buffer[current_item_index];
	}
	
	// Exclude currently last item if its weight exceeds knapsack space and recurse further.
	if((u32)items[n-1].weight > knapsack_space) {
		buffer[current_item_index] = solve1(items, n-1, knapsack_space, buffer, total_knapsack_space);
		return buffer[current_item_index];
	}

	// Go through cases where we exclude currently last item and where we include it.
	// If we include it, then we count its value and weight and reduce current knapsack space.
	
	// We pick the path that gives maximum value for the current step (local computational maximum).
	Item value_excluded = solve1(items, n-1, knapsack_space, buffer, total_knapsack_space);
	Item value_included = solve1(items, n-1, knapsack_space - (u32)items[n-1].weight, buffer, total_knapsack_space);

	value_included.weight += (u32)items[n-1].weight;
	value_included.value  += (u32)items[n-1].value;

	if(value_excluded.value > value_included.value) {
		buffer[current_item_index] = value_excluded;
	}
	else {
		buffer[current_item_index] = value_included;
	}

	return buffer[current_item_index];
}

Item recursive_solution_with_2D_buffer_and_integer_weights(Item* items, u32 n, u32 knapsack_space) {
	// knapsack_space + 1 because we need weight range [0, knapsack_space].
	// n used instead of (n+1) because we handle n = 0 case explicitly with if.
	u32 buffer_size = n*(knapsack_space + 1);
	Item* buffer = malloc(buffer_size*sizeof(Item));

	if(!buffer) {
		printf("Memory allocation failed.\n");
		return (Item){0, 0};
	}

	for(u32 i = 0; i < buffer_size; ++i) {
		buffer[i].value = -1;
	}
	
	Item result = solve1(items, n, knapsack_space, buffer, knapsack_space);
	
	free(buffer);
	return result;
}

// =========================================================================================================
// ITERATIVE SOLUTION WITH 2D BUFFER (ITEM WEIGHTS AND KNAPSACK WEIGHT MUST BE INTEGERS IN THIS APPROACH)
//
// Time complexity  : O(n*total_knapsack_space)
// Space complexity : O(n*total_knapsack_space)
// =========================================================================================================

// This approach has no recursion stack frame drawbacks. Overall approach is the same as with recursion,
// in the sense that we move through the items and we are making a choice of inclusion by taking the
// maximum value at the current step ie. we are making local maximum steps in hopes that they lead us
// closer to the global maximum.

// Table value at location (i, w) represents maximum accumulated value of (i) items (some of them are
// included and some are not), given that the current knapsack weight is at maximum (w).
// When deciding if we will include item (i) for some knapsack weight (w), we check the maximum value
// accumulated from the previous items for that (w) in the knapsack (this is at location (i-1, w)). This
// is the value without current item, meaning that if we update the table like this T(i, w) = T(i-1, w),
// then we effectively chose to not include the current item. If, on the other hand, we want to include
// the item, then that means that the location (i, w) must represent value after counting that item. Because
// of this, we update the table like this T(i, w) = T(i-1, w - weight(i)) + value(i) ie. we are saying
// that we will end up with the weight (w) if we were previously with the weight (w - weight(i)) and
// then we added the current item whose weight is weight(i). We are also saying that the new value in
// this case is the previous one at the mentioned postion plus the value of our current item.

// Since our overall tactic is to always go for the local maximum value, we make a choice between these
// two cases by taking the bigger value, and then we place it in the table.

// Final result is stored as the final element in the table and the interpretation of this position
// is that it is the maximum value accumulated by (i) items such that the total weight in knapsack
// is at most (w).

Item iterative_solution_with_2D_buffer_and_integer_weights(Item* items, u32 n, u32 knapsack_space) {
	u32 buffer_size = (n)*(knapsack_space + 1);
	
	// Set all positions in the buffer to zero so that we don't need to worry.
	Item* buffer = calloc(buffer_size, sizeof(Item));

	if(!buffer) {
		printf("Memory allocation failed.\n");
		return (Item){0, 0};
	}

	// Fill the whole first row with first item value so that the first iteration in
	// the loop below can immediately reference those values.
	// We do this to avoid making an if/else exception within the loop that will just
	// handle this initial case.
	for(u32 i = items[0].weight; i < knapsack_space + 1; ++i) {
		buffer[i].value = (u32)items[0].value;
		buffer[i].weight = (u32)items[0].weight;
	}

	Item value_included = {0,0};

// This macro just calculates element index within our buffer when we supply indices as
// we would in the case of a matrix.
#define buffer_index(i, w) ((i)*(knapsack_space + 1) + (w))

	// First item is already handled, so we start with 1.
	for(u32 i = 1; i < n; ++i) {
		// Iterate over all knapsack weights that are lower than the current item weight
		// and just set all positions (i, w) to previous values from (i-1, w), since
		// we know that we can't include the current item (because of its weight).
		for(u32 w = 0; w < (u32)items[i].weight && w <= knapsack_space; ++w) {
			buffer[buffer_index(i, w)] = buffer[buffer_index(i-1, w)];
		}
		
		// Iterate over all possible knapsack weights for which we can include the current item.
		// These are all knapsack weights that are >= current item weight.
		// For all locations (i, w) for these weights, make a choice to include or exclude the item.
		for(u32 w = knapsack_space; w >= (u32)items[i].weight && w > 0; --w) {
			// This is the value when the current item is included.
			// We get this value by recognizing that it is the value of the current item plus the value
			// that was accumulated from the previous items, given that their weight was (w - weight(i))
			// ie. if we add our item to that weight, we get (w). In other words, we add current item value
			// to whatever value is at (i-1, w - weight(i)).
			value_included.value = buffer[buffer_index(i-1, w - (u32)items[i].weight)].value + (u32)items[i].value;
			value_included.weight = buffer[buffer_index(i-1, w - (u32)items[i].weight)].weight + (u32)items[i].weight;

			// If the value is greater when the item is included then choose that, otherwise choose
			// the previous value for that weight (w).
			if(value_included.value > buffer[buffer_index(i-1, w)].value) {
				buffer[buffer_index(i, w)] = value_included;
			}
			else {
				buffer[buffer_index(i, w)] = buffer[buffer_index(i-1, w)];
			}
		}
	}

	// Final result is at the last location in the buffer since the interpretation of that
	// location is that it holds the maximum value accumulated by (n) items, such that
	// their weight is at most (w), which is what we are looking for.
	Item result = buffer[buffer_index(n-1, knapsack_space)];
	
#undef buffer_index
	
	free(buffer);
	return result;
}

// =========================================================================================================
// ITERATIVE SOLUTION WITH 1D BUFFER (ITEM WEIGHTS AND KNAPSACK WEIGHT MUST BE INTEGERS IN THIS APPROACH)
//
// Time complexity  : O(n*total_knapsack_space)
// Space complexity : O(total_knapsack_space)
// =========================================================================================================

// This solution is like the previous one, but with the lower space complexity, which is now linear
// instead of quadratic.

// This is an example of the 2D buffer used in the previous iterative solution for 20 randomly
// generated items with the seed 1234 and knapsack of size 20.

//   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20 
// ====================================================================================
//   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 
//   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 
//   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 
//   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 
//   0   0   0   0   0   0   0   0   0   0   0   0  40  40  40  40  40  40  40  40  40 
//   0   0   0   0   0   0   0   0   0   0   0   0  40  40  40  40  40  40  40  40  40 
//   0   0   0   0   0   0   0   0   0   0   0   0  40  40  40  40  40  40  40  40  40 
//   0   0   0   0   0   0   0   0   0   0   0   0  40  40  40  40  40  40  40  40  40 
//   0   0   0   0   0   0   0   0   0   0   0   0  40  40  40  40  40  40  40  40  67 
//   0   0   0  35  35  35  35  35  35  35  35  35  40  40  40  75  75  75  75  75  75 
//   0   0   0  35  35  35  35  35  35  35  35  35  40  40  40  75  75  75  75  75  75 
//   0   0   0  35  35  35  35  35  35  95  95  95 130 130 130 130 130 130 130 130 130 
//   0   0   0  35  35  35  35  35  35  95  95  95 130 130 130 130 130 130 130 130 130 
//   0   0   0  35  35  35  35  35  35  95  95  95 130 130 130 130 130 130 130 130 130 
//   0   0   0  35  35  35  35  35  35  95  95  95 130 130 130 130 130 130 130 130 130 
//   0   0   0  35  35  35  35  35  35  95  95  95 130 130 130 130 130 130 130 130 130 
//   0   0   0  35  35  35  35  35  35  95  95  95 130 130 130 130 130 130 130 130 130 
//   0   0   0  35  35  35  35  35  35  95  95  95 130 130 130 130 130 130 130 130 130 
//   0   0   0  35  35  35  35  35  35  95  95  95 130 130 130 130 130 130 130 130 130 
//   0   0   0  35  35  35  35  35  35  95  95  95 130 130 130 130 130 130 130 130 130

// One important thing to notice in the previous algorithm is that in order to fill up the next row,
// we just need values from the previous row, not from the whole buffer.
// This means that we need at most 2 1D arrays. One that is currently updated and the other one that
// is currently read.

// But, we can even get to only one array of size of knapsack space, by noticing that if we calculate
// new values from the right to left, like we did in the previous algorithm, there will be no conflict
// during the update of new values, even if they are in the same row.
// This happens because in order to calculate the value at position (w), we just need to reference some
// position that is lower than that (w). Precise position will be (w) - weight(i) ie. weight for
// which we are calculating new value minus the current item weight.

Item iterative_solution_with_1D_buffer_and_integer_weights(Item* items, u32 n, u32 knapsack_space) {
 	u32 buffer_size = knapsack_space + 1;

	// Set all positions in the buffer to zero so that we don't need to worry.
	Item* buffer = calloc(buffer_size, sizeof(Item));

	Item value_included = {0, 0};

	// Go through all items.
	for(u32 i = 0; i < n; ++i) {
		// Go through all knapsack weights that can include current item.
		for(u32 w = knapsack_space; w >= (u32)items[i].weight && w > 0; --w) {
			// Calculate relevant properties in the case of including the item.
			value_included.value = buffer[w - (u32)items[i].weight].value + (u32)items[i].value;
			value_included.weight = buffer[w - (u32)items[i].weight].weight + (u32)items[i].weight;

			// If value is greater when item is included then pick that value, otherwise do nothing
			// since the old value is already there because we are using the same buffer.
			if(value_included.value > buffer[w].value) {
				buffer[w] = value_included;
			}
		}
	}

	Item result = buffer[knapsack_space];

	free(buffer);
	return result;
}

// =========================================================================================================
// SIMULATED ANNEALING APPROXIMATE SOLUTION
//
// Time complexity  : O(iteration_count)
// Space complexity : O(number of bits needed to keep state of included items)
// =========================================================================================================

// This is an approximate approach that allows us to solve the problem in linear time in the number of
// iterations.

// In order to solve the problem this way, we view it as a problem of finding a path from the starting point
// to some ideal final point in phase space. Phase space is just a space that contains all possible states
// of the problem. In our case, a state is an array/vector whose element at index "i" tells us if an
// item with index "i" is included in the knapsack in that state. Thus, our phase space has 2^n points,
// where n is the number of items.

// Since we don't know what the optimal solution is for the given input, in order to move towards it, we
// need to be able to tell how good some state is. We can just say that a state is better if it has a
// larger value, as long as its weight doesn't exceed knapsack space. This allows us to also compare two
// states ie. it allows us to decide whether we should move to the next state. In other words, it allows
// us to build path from the starting point towards some optimal point.

// We start at some random state. We choose a possible next state by making a small local step in phase
// space. In this implementation, that step is made by just flipping one state element. After this, we
// check if we are in a better state (state with larger value). If the state is better, then we accept
// it immediately. If it is worse, then we accept it with some probability that should decay as we are
// traveling from start to finish through the phase space. Reason for accepting worse solutions with
// some probability, instead of just rejecting them, is that we don't want to end up is some local
// optimum from which we can't escape. On the other hand, we want to converge to a decent solution as
// we move further, which is why we lower acceptance probability with time. Intuitively, this strategy
// means that we are willing to take risks while the phase space is still relatively unknown to us. But
// as we explore it further, we will stumble upon paths that seem promising and at that point we will be
// less willing to significantly diverge from those paths.

// We control probability decay by using a sequnce of scaled botzmann distributions ie. functions of the
// form (A*e^(-cost_diff/T)). When we lower "T", overall value will be smaller, meaning that we can lower
// this parameter, called temperature, if we want lower probability. "A" just allows us to experiment.
// Cost difference is a difference between the values of previous and current state and should be
// positive, because if it is negative, that means that we found a better state and should certainly
// choose it and in this case, we don't need distribution to tell us the probability of choice.
// When the cost difference is higher, probability will be lower, meaning that we are less sure of
// making a much worse state choice.

// In short, we just need to keep lowering temperature in each epoch (epoch is some number of
// iterations). This will basically choose a particular Boltzmann distribution for specific epoch,
// which we will then sample with cost differences. Every next epoch, a new distribution is chosen
// that has overall smaller probabilities than the previous one.

Item simulated_annealing_solution(Item* items, u32 n, f32 knapsack_space, Simulated_Annealing_Parameters SAP) {
	// Keeps track of which items are currently included.
	// Effectively represents one point in the phase space of the system.
	// Here, we use byte to store one bit of information, which is a waste, and a better implementation
	// would fully utilize memory space.
	u8* state = calloc(n, sizeof(u8));

	Item previous_cost = {0, 0};
	Item cost = {0, 0};

	// Randomly pick items for initial state, such that maximum knapsack space is not exceeded.
	for(u32 i = 0; i < n; ++i) {
		state[i] = random_u8();
		if(state[i]) {
			if(cost.weight + items[i].weight > knapsack_space) {
				state[i] = 0;
			}
			else {
				cost.value += items[i].value;
				cost.weight += items[i].weight;
			}
		}
	}

	// Find the maximum value, which we will use to scale cost difference before using
	// it with Boltzmann distribution.
	f32 max_value = items[0].value;
	for(u32 i = 0; i < n; ++i) {
		if(items[i].value > max_value) {
			max_value = items[i].value;
		}
	}

	f32 mapped_cost_difference = 0;
	f32 acceptance_probability = 0;
	while(SAP.iteration_count--) {
		// Pick random item index.
		u32 random_item_index = random_int_in_range_exclusive(0, n);

		// Flip the state for picked item.
		state[random_item_index] ^= 1;

		// This is used in the case of rejection of current change.
		previous_cost = cost;

		// If the item is now included.
		if(state[random_item_index]) {
			cost.value += items[random_item_index].value;
			cost.weight += items[random_item_index].weight;

			// If the weight was exceeded.
			if(cost.weight > knapsack_space) {
				cost = previous_cost;
				state[random_item_index] ^= 1;
				continue;
			}
		}
		else {
			cost.value -= items[random_item_index].value;
			cost.weight -= items[random_item_index].weight;
		}

		// If we are in the worse state after making a change.
		if(cost.value < previous_cost.value) {
			// Be careful here because cost difference should be positive. Since we know that within this
			// condition current value is smaller, we force this by subtracting it from previous one and not the
			// other way around. Reason for this is that Boltzmann distribution has minus in the exponent.
			mapped_cost_difference = (previous_cost.value - cost.value) / (max_value) * SAP.cost_scalar;
			acceptance_probability = boltzmann_distribution(SAP.boltzmann_scalar, mapped_cost_difference, SAP.temperature);

			// In this case we reject the change.
			if(random_f32() > acceptance_probability) {
				cost = previous_cost;
				state[random_item_index] ^= 1;
			}
		}

		// Adjust temperature every epoch_size iterations and do it if the temperature is still
		// larger than minimum temperature.
		if(SAP.iteration_count % SAP.epoch_size == 0 && SAP.temperature > SAP.minimal_temperature) {
			SAP.temperature *= SAP.temperature_scalar;
		}
	}

	free(state);
	return cost;
}

// =========================================================================================================

#include <time.h>

int main(void) {
	srand(time(0));
	
	u32 n = 100;
	f32 knapsack_space = 5000;
	f32 max_item_weight = 1000;
	f32 max_item_value = 1000;

	Item* items = generate_random_items(n, max_item_weight, max_item_value);

	Simulated_Annealing_Parameters SAP = {
		.iteration_count = 10000,
		.epoch_size = 10,
		.temperature = 10,
		.temperature_scalar = 0.95,
		.minimal_temperature = 0.1,
		.cost_scalar = 6,
		.boltzmann_scalar = 1
	};
	
	if(items) {
		Item deterministic_result = {0, 0};
		Item stochastic_result = {0, 0};
		// print_items(items, n);
		// deterministic_result = recursive_solution(items, n, knapsack_space);
		// deterministic_result = recursive_solution_with_2D_buffer_and_integer_weights(items, n, (u32)knapsack_space);
		// deterministic_result = iterative_solution_with_2D_buffer_and_integer_weights(items, n, (u32)knapsack_space);
		deterministic_result = iterative_solution_with_1D_buffer_and_integer_weights(items, n, (u32)knapsack_space);
		
		printf("DETERMINISTIC:\n");
		printf("\tMaximum weight: %lf\n", deterministic_result.weight);
		printf("\tMaximum value : %lf\n", deterministic_result.value);
		
		stochastic_result = simulated_annealing_solution(items, n, knapsack_space, SAP);
		Item max_stochastic_result = stochastic_result;
		for(u32 i = 0; i < 20; ++i) {
			srand(time(0));
			stochastic_result = simulated_annealing_solution(items, n, knapsack_space, SAP);
			if(stochastic_result.value > max_stochastic_result.value) {
				max_stochastic_result = stochastic_result;
			}
		}
		printf("STOCHASTIC:\n");
		printf("\tMaximum weight: %lf\n", max_stochastic_result.weight);
		printf("\tMaximum value : %lf\n", max_stochastic_result.value);
	}
	
	return 0;
}

