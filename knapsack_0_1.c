#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define IGNORE(param) (void)(param)

typedef unsigned char u8;
typedef unsigned int u32;
typedef float f32;

typedef struct Item Item;
struct Item {
	f32 weight;
	f32 value;
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
		return items;
	}
	else {
		return 0;
	}
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

f32 boltzmann_distribution(f32 cost_difference, f32 temperature) {
	return exp(-cost_difference/temperature);
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
// Time complexity  :
// Space complexity :
// =========================================================================================================

Item simulated_annealing_solution(Item* items, u32 n, f32 knapsack_space) {
	u8* state = calloc(n, sizeof(u8));

	f32 temperature = 1;

	Item previous_cost = {0, 0}
	Item cost = {0, 0};

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

	while(temperature > 1e-10) {
		u32 random_item_index = random_int_in_range_exclusive(0, n);

		// We are just flipping the state for random item.

		previous_cost = cost;
		if(state[random_item_index]) {
			cost.value -= items[random_item_index];
			cost.weigth -= items[random_item_index];
		}
		else {
			cost.value += items[random_item_index];
			cost.weigth += items[random_item_index];
		}

		// TODO(stekap): Maybe add rescaling for cost difference.

		// We count this case as probability of 1.
		if(cost.value - previous_cost < 0) {
			
		}
		else {
			f32 probability = boltzmann_distribution(cost.value - previous_cost, temperature);

			// In this case we accept the solution.
			if(random_f32() < probability) {
				// Accept.
			}
			else {
				// Reject.
			}
		}

		temperature *= 0.95;
	}

	free(state);
	return (Item){0, 0};
}

// =========================================================================================================

int main(void) {
	u32 seed = 1234;
	srand(seed);
	u32 n = 100;
	f32 knapsack_space = 5000;
	
	Item* items = generate_random_items(n, 100, 100);
	if(items) {
		//print_items(items, n);
		//Item result = recursive_solution(items, n, knapsack_space);
		//Item result = recursive_solution_with_2D_buffer_and_integer_weights(items, n, (u32)knapsack_space);
		//Item result = iterative_solution_with_2D_buffer_and_integer_weights(items, n, (u32)knapsack_space);
		//Item result = iterative_solution_with_1D_buffer_and_integer_weights(items, n, (u32)knapsack_space);
		Item result = simulated_annealing_solution(items, n, knapsack_space);
		printf("Maximum weight: %lf\n", result.weight);
		printf("Maximum value : %lf\n", result.value);
	}
	
	return 0;
}

