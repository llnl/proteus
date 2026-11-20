target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; Every object is a local allocation. All called functions are defined here;
; no globals, opaque effects, atomics, or synchronization are involved.
; Each root initializes target to 1, then invokes a helper that writes 42
; through another object's stored pointer before reading target.
define internal i32 @read_target(ptr readonly %target) nosync memory(argmem: read) {
  %value = load i32, ptr %target
  ret i32 %value
}

; A pointer slot passed by address: overwrite(&slot), where slot == &target.
; Reading the slot still allows writing through the pointer loaded from it.
define internal void @pointer_slot.write(ptr readonly %slot) nosync {
  %destination = load ptr, ptr %slot
  store i32 42, ptr %destination
  ret void
}
define internal i32 @pointer_slot.helper(ptr %target, ptr %slot) nosync {
  call void @pointer_slot.write(ptr %slot)
  %value = call i32 @read_target(ptr %target)
  ret i32 %value
}
define i32 @pointer_slot.root() nosync {
  %target = alloca i32
  %slot = alloca ptr
  store i32 1, ptr %target
  store ptr %target, ptr %slot
  ; The pointer conversion must still expose the store through %target above.
  %bits = ptrtoint ptr %target to i64
  %converted = inttoptr i64 %bits to ptr
  %value = call i32 @pointer_slot.helper(ptr %converted, ptr %slot)
  ret i32 %value
}

; Two stored-pointer hops: outer -> inner -> target. Checking the outer
; allocation, or just one level of its fields, is insufficient.
define internal void @nested_holder.write(ptr %outer) nosync {
  %inner = load ptr, ptr %outer
  %destination = load ptr, ptr %inner
  store i32 42, ptr %destination
  ret void
}
define internal i32 @nested_holder.helper(ptr %target, ptr %outer) nosync {
  call void @nested_holder.write(ptr %outer)
  %value = call i32 @read_target(ptr %target)
  ret i32 %value
}
define i32 @nested_holder.root() nosync {
  %target = alloca i32
  %inner = alloca ptr
  %outer = alloca ptr
  store i32 1, ptr %target
  store ptr %target, ptr %inner
  store ptr %inner, ptr %outer
  %value = call i32 @nested_holder.helper(ptr %target, ptr %outer)
  ret i32 %value
}

; The selected array entry holds target; the first entry holds unrelated
; storage. A nonzero field offset must not hide the stored pointer.
define internal void @pointer_array.write(ptr %table) nosync {
  %entry = getelementptr [2 x ptr], ptr %table, i64 0, i64 1
  %destination = load ptr, ptr %entry
  store i32 42, ptr %destination
  ret void
}
define internal i32 @pointer_array.helper(ptr %target, ptr %table) nosync {
  call void @pointer_array.write(ptr %table)
  %value = call i32 @read_target(ptr %target)
  ret i32 %value
}
define i32 @pointer_array.root() nosync {
  %target = alloca i32
  %other = alloca i32
  %table = alloca [2 x ptr]
  %entry = getelementptr [2 x ptr], ptr %table, i64 0, i64 1
  store i32 1, ptr %target
  store i32 0, ptr %other
  store ptr %other, ptr %table
  store ptr %target, ptr %entry
  %value = call i32 @pointer_array.helper(ptr %target, ptr %table)
  ret i32 %value
}

; A getter returns the stored pointer across another call boundary, and its
; caller writes through the returned value.
define internal ptr @returned_pointer.get(ptr readonly %slot) nosync memory(argmem: read) {
  %destination = load ptr, ptr %slot
  ret ptr %destination
}
define internal void @returned_pointer.write(ptr %slot) nosync {
  %destination = call ptr @returned_pointer.get(ptr %slot)
  store i32 42, ptr %destination
  ret void
}
define internal i32 @returned_pointer.helper(ptr %target, ptr %slot) nosync {
  call void @returned_pointer.write(ptr %slot)
  %value = call i32 @read_target(ptr %target)
  ret i32 %value
}
define i32 @returned_pointer.root() nosync {
  %target = alloca i32
  %slot = alloca ptr
  store i32 1, ptr %target
  store ptr %target, ptr %slot
  %value = call i32 @returned_pointer.helper(ptr %target, ptr %slot)
  ret i32 %value
}
