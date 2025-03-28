import pstats

p = pstats.Stats('profiling_results.out')
p.strip_dirs().sort_stats('time').print_stats()
