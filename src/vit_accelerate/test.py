import paddle.profiler as profiler
with profiler.Profiler(
targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
scheduler = profiler.make_scheduler(closed=1, ready=1, record=3, repeat=3),
on_trace_ready = profiler.export_chrome_tracing('./log')) as p:
    for iter in range(10):
        #train()
        p.step()