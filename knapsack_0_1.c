#include <stdio.h>
#include <stdlib.h>

typedef unsigned int u32;
typedef unsigned long long u64;
typedef double f64;

typedef struct Item Item;
struct Item {
	f64 weight;
	f64 value;
};

Item* generate_random_items(u64 n, u32 seed, f64 max_item_weight, f64 max_item_value) {
	srand(seed);
	Item* items = malloc(n * sizeof(Item));
	if(items) {
		for(u64 i = 0; i < n; ++i) {
			items[i].weight = (f64)rand()/(f64)RAND_MAX * max_item_weight;
			items[i].value = (f64)rand()/(f64)RAND_MAX * max_item_value;
		}
		return items;
	}
	else {
		return 0;
	}
}

void print_items(Item* items, u64 n) {
	for(u64 i = 0; i < n; ++i) {
		printf("( %9.6lf, %9.6lf )\n", items[i].weight, items[i].value);
	}
}

// Returns the maximum value that can be achieved while not exceeding knapsack space.
// Also returns maximum reached weight.

// This algorithm moves recursively from the last item in a greedy fashion ie. when
// deciding whether to include current item or not, it will always take the route
// that gives it maximum value at the current step (local maximum).

// This is not efficient solution, since it does redundant computations.
Item recursive_solution(Item* items, u64 n, f64 knapsack_space) {
	// If no more space or no more items.
	if(knapsack_space == 0 || n == 0) {
		return (Item){0, 0};
	}

	// Exclude currently last item if its weight exceeds knapsack space.
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

int main(void) {
	u32 seed = 1234;
	u64 n = 10;
	f64 knapsack_space = 150;
	(void)knapsack_space;
	
	Item* items = generate_random_items(n, seed, 100, 100);
	if(items) {
		print_items(items, n);
		Item result = recursive_solution(items, n, knapsack_space);
		printf("Maximum weight: %lf\n", result.weight);
		printf("Maximum value : %lf\n", result.value);
	}
	
	return 0;
}
