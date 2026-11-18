target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @consume(ptr readonly) memory(read)
declare void @opaque_capture(ptr readonly) memory(inaccessiblemem: write)

@unrelated_global = internal global i32 0, align 4
@device_format = addrspace(4) constant [4 x i8] c"%d\00\00"

define internal void @write_global(ptr %p) {
  store i32 7, ptr %p, align 4
  ret void
}

define internal void @use_closure(ptr %p) {
  %value = load i32, ptr %p, align 4
  ret void
}

define internal void @simple_use_closure(ptr %p) {
  %value = load i32, ptr %p, align 4
  ret void
}

define internal void @closure_wrapper(ptr %p) {
  call void @use_closure(ptr %p)
  ret void
}

; A direct pointer into global storage cannot be the registered closure.
; The call may write memory, but its global target is disjoint from %target.
define internal void @global_arg_helper(ptr %target) {
  call void @write_global(ptr @unrelated_global)
  call void @simple_use_closure(ptr %target)
  ret void
}

; Match the AMDGPU address-space shape from shared_array: the closure arrives
; in global address space and is cast to generic before the wrapper call; the
; unrelated write receives an address-space-cast global format string.
define amdgpu_kernel void @device_global_arg(ptr addrspace(1) %closure) {
  %generic = addrspacecast ptr addrspace(1) %closure to ptr
  call void @write_global(ptr addrspacecast (ptr addrspace(4) @device_format to ptr))
  call void @opaque_capture(ptr %generic)
  call void @closure_wrapper(ptr %generic)
  ret void
}

; Same optnone spill shape as lambda_alias_argument. The test starts collection
; at %target; reaching the copy requires the compressed %target -> %alias edge.
define internal void @helper(ptr %target, ptr %alias, ptr %replacement) {
  %target.slot = alloca ptr
  %alias.slot = alloca ptr
  store ptr %target, ptr %target.slot
  store ptr %alias, ptr %alias.slot
  %writer = load ptr, ptr %alias.slot
  call void @llvm.memcpy.p0.p0.i64(ptr %writer, ptr %replacement, i64 4, i1 false)
  %body = load ptr, ptr %target.slot
  call void @consume(ptr %body)
  ret void
}

define void @root() {
  %initial = alloca i32
  %replacement = alloca i32
  %other = alloca i32
  call void @helper(ptr %initial, ptr %initial, ptr %replacement)
  call void @helper(ptr %replacement, ptr %initial, ptr %other)
  ret void
}

; One inner call instruction, two different outer contexts.
define internal void @nested(ptr %target, ptr %alias) {
  call void @consume(ptr %target)
  ret void
}
define internal void @middle(ptr %a, ptr %b) {
  call void @nested(ptr %a, ptr %b)
  ret void
}
define void @nested.root() {
  %a = alloca i32
  %b = alloca i32
  call void @middle(ptr %a, ptr %a)
  call void @middle(ptr %a, ptr %b)
  ret void
}

define internal void @displaced(ptr %target, ptr %alias) {
  ret void
}
define void @displaced.root() {
  %storage = alloca [4 x i32]
  %field = getelementptr [4 x i32], ptr %storage, i64 0, i64 1
  call void @displaced(ptr %field, ptr %storage)
  ret void
}

; Distinct externally supplied pointers must not become a NoAlias proof.
define void @incoming(ptr %target, ptr %alias) {
  ret void
}

define internal void @reloaded(ptr %target, ptr %alias) {
  ret void
}
define void @reload.root() {
  %a = alloca i32
  %b = alloca i32
  %slot = alloca ptr
  store ptr %a, ptr %slot
  %before = load ptr, ptr %slot
  call void @reloaded(ptr %a, ptr %before)
  store ptr %b, ptr %slot
  %after = load ptr, ptr %slot
  call void @reloaded(ptr %a, ptr %after)
  ret void
}

; A local alias cannot make a later, unreachable-from-the-use write clobber it.
define internal void @late(ptr %target, ptr %alias) {
  call void @consume(ptr %target)
  store i32 7, ptr %alias
  ret void
}
define void @late.root() {
  %a = alloca i32
  call void @late(ptr %a, ptr %a)
  ret void
}

; Call-chain cycles must terminate conservatively.
define internal void @recursive(ptr %a, ptr %b) {
  call void @recursive(ptr %a, ptr %b)
  ret void
}

; The function body is deliberately unavailable. Its LLVM effects make the
; pointer arguments a complete set of possible write roots, but only the
; caller-dependent graph distinguishes output from target inside output_helper.
declare void @write_output(ptr writeonly) nosync memory(argmem: write)
define internal void @output_helper(ptr %target, ptr %output) {
  call void @write_output(ptr %output)
  call void @consume(ptr %target)
  ret void
}
define void @output.root() {
  %a = alloca i32
  %b = alloca i32
  call void @output_helper(ptr %a, ptr %b)
  call void @output_helper(ptr %a, ptr %a)
  ret void
}

; No explicit pointer arguments does not prove disconnection when the effect
; contract allows writes through other inputs.
declare void @opaque_effect() nosync
define void @opaque_helper(ptr %target) {
  call void @opaque_effect()
  call void @consume(ptr %target)
  ret void
}

; A complete, disconnected graph must not require the incoming target to
; resolve to an alloca. No nosync/argmem-only contract is supplied for the
; defined utility: the body and the absence of a connection are available.
define internal void @defined_local_effect() {
  %scratch = alloca i32
  store i32 42, ptr %scratch
  ret void
}
define void @incoming_disconnected(ptr %target) {
  %bits = ptrtoint ptr %target to i64
  %converted = inttoptr i64 %bits to ptr
  call void @defined_local_effect()
  call void @consume(ptr %converted)
  ret void
}
