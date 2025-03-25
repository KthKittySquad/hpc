import pstats

# Load the stats file
stats = pstats.Stats("profile.stats")
# Sort by cumulative time and print the top 20 functions
stats.sort_stats("tottime").print_stats(5)
