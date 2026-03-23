DISPATCH_DYLIB = libexpert_reader.dylib
SERIAL_DYLIB   = libexpert_reader_serial.dylib
OUTDIR         = ../.run

all: $(DISPATCH_DYLIB) $(SERIAL_DYLIB)

# Dispatch build: expert_io.c (pread) + expert_mem.c (alloc_slab, free_slab,
# alloc_buffer, free_buffer, copy_experts_multi).
# libdispatch is part of libSystem on macOS — no extra -framework needed.
$(DISPATCH_DYLIB): expert_io.c expert_mem.c
	clang -O2 -fPIC -shared $^ -o $@

# Serial build: only implements read_component_batch / read_component_batches.
# Does NOT have alloc_slab, free_slab, copy_experts_multi or the slots API.
$(SERIAL_DYLIB): expert_reader_serial.c
	clang -O2 -fPIC -shared $< -o $@

install: all
	cp $(DISPATCH_DYLIB) $(SERIAL_DYLIB) $(OUTDIR)/

clean:
	rm -f $(DISPATCH_DYLIB) $(SERIAL_DYLIB)
