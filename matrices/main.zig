const std = @import("std");
const mem = std.mem;

pub fn Matrix(comptime T: type) type {
    return struct {
        m: [*]Vec,
        stride: u16,
        rows: u16,
        cols: u16,

        pub const Elem = T;
        pub const Vec = @Vector(std.simd.suggestVectorSize(Elem) orelse 1, Elem);

        fn vecCount(rows: usize, cols: usize) usize { return rows*@divExact(cols, length(Vec)); }

        pub fn initUndefined(allocator: mem.Allocator, rows: u16, cols: u16) !@This() {
            const m = try allocator.alloc(Vec, vecCount(rows, cols));
            return of(rows, cols, m);
        }

        pub fn deinit(a: @This(), allocator: mem.Allocator) void {
            std.debug.assert(a.cols == a.stride*length(Vec));
            allocator.free(a.items());
        }

        pub inline fn of(rows: u16, cols: u16, m: []Vec) @This() {
            std.debug.assert(vecCount(rows, cols) <= m.len);
            const stride = @divExact(cols, length(Vec));
            return .{ .m = m.ptr, .stride = stride, .rows = rows, .cols = cols };
        }

        pub fn copy(a: @This(), m: []Vec) @This() {
            const res = of(a.rows, a.cols, m);
            for (0..a.rows) |row| mem.copy(Vec, res.rowAt(row), a.rowAt(row));
            return res;
        }

        pub fn copyTransposed(a: @This(), m: []Vec) @This() {
            const res = of(a.rows, a.cols, m);
            for (0..a.rows) |row| {
                for (a.rowAt(row), 0..) |v, vcol| {
                    inline for (comptime 0..length(Vec)) |k| {
                        const col = vcol*length(Vec) + k;
                        res.set(col, row, v[k]);
                    }
                }
            }
            return res;
        }

        pub inline fn get(a: @This(), row: usize, col: usize) Elem {
            return a.m[row*a.stride + col/length(Vec)][col%length(Vec)];
        }

        pub inline fn set(a: @This(), row: usize, col: usize, v: Elem) void {
            a.m[row*a.stride + col/length(Vec)][col%length(Vec)] = v;
        }

        pub inline fn rowAt(a: @This(), row: usize) []Vec {
            const first = row*a.stride;
            return a.m[first .. first + @divExact(a.cols, length(Vec))];
        }

        pub inline fn items(a: @This()) []Vec {
            return a.m[0 .. @as(usize, a.stride)*a.rows];
        }

        pub inline fn setAll(a: @This(), v: Elem) @This() {
            for (0..a.rows) |row| @memset(a.rowAt(row), @splat(length(Vec), v));
            return a;
        }

        pub fn add(res: @This(), a: @This()) @This() {
            std.debug.assert(res.rows == a.rows and res.cols == a.cols);
            for (0..a.rows) |row| {
                for (res.rowAt(row), a.rowAt(row)) |*r, v| r.* += v;
            }
            return res;
        }

        pub fn sub(res: @This(), a: @This()) @This() {
            std.debug.assert(res.rows == a.rows and res.cols == a.cols);
            for (0..a.rows) |row| {
                for (res.rowAt(row), a.rowAt(row)) |*r, v| r.* -= v;
            }
            return res;
        }

        pub fn subMatrix(a: @This(), row: usize, col: usize, rows: u16, cols: u16) @This() {
            return .{
                .m = a.m + row*a.stride + @divExact(col, length(Vec)),
                .stride = a.stride,
                .rows = rows,
                .cols = cols,
            };
        }

        pub fn addProduct(res: @This(), a: @This(), b: @This(), temp: []Vec) void {
            std.debug.assert(res.rows == a.rows and res.cols == b.cols and a.cols == b.rows);
            if (res.rows <= length(Vec)*16) {
                return res.addProductRhsTransposed(a, b.copyTransposed(temp));
            }
            const rowMid = @divExact(res.rows, 2);
            const mulMid = @divExact(a.cols, 2);
            const colMid = @divExact(res.cols, 2);

            const r11 = res.subMatrix(0, 0, rowMid, colMid);
            const r12 = res.subMatrix(0, colMid, rowMid, colMid);
            const r21 = res.subMatrix(rowMid, 0, rowMid, colMid);
            const r22 = res.subMatrix(rowMid, colMid, rowMid, colMid);

            const a11 = a.subMatrix(0, 0, rowMid, mulMid);
            const a12 = a.subMatrix(0, mulMid, rowMid, mulMid);
            const a21 = a.subMatrix(rowMid, 0, rowMid, mulMid);
            const a22 = a.subMatrix(rowMid, mulMid, rowMid, mulMid);

            const b11 = b.subMatrix(0, 0, mulMid, colMid);
            const b12 = b.subMatrix(0, colMid, mulMid, colMid);
            const b21 = b.subMatrix(mulMid, 0, mulMid, colMid);
            const b22 = b.subMatrix(mulMid, colMid, mulMid, colMid);

            const atemp = temp[vecCount(rowMid, colMid)..];
            const btemp = atemp[vecCount(rowMid, mulMid)..];
            const ttemp = btemp[vecCount(mulMid, colMid)..];

            const tr = of(rowMid, colMid, temp);
            tr.setAll(0).addProduct(a21.copy(atemp).sub(a11), b12.copy(btemp).sub(b22), ttemp);
            _ = r21.add(tr);
            _ = r22.add(tr);

            tr.setAll(0).addProduct(a21.copy(atemp).add(a22), b12.copy(btemp).sub(b11), ttemp);
            _ = r12.add(tr);
            _ = r22.add(tr);

            tr.setAll(0).addProduct(a11, b11, ttemp);
            _ = r11.add(tr);
            r11.addProduct(a12, b21, ttemp);
            tr.addProduct(a21.copy(atemp).add(a22).sub(a11), b11.copy(btemp).add(b22).sub(b12), ttemp);
            _ = r12.add(tr);
            _ = r21.add(tr);
            _ = r22.add(tr);

            r12.addProduct(a11.copy(atemp).add(a12).sub(a21).sub(a22), b22, ttemp);
            r21.addProduct(a22, b21.copy(btemp).add(b12).sub(b11).sub(b22), ttemp);
        }

        pub fn addProductRhsTransposed(res: @This(), a: @This(), bT: @This()) void {
            std.debug.assert(res.rows == a.rows and res.cols == bT.rows and a.cols == bT.cols);
            for (0..res.cols) |col| {
                for (0..res.rows) |row| {
                    var sum = @splat(length(Vec), @as(Elem, 0));
                    for (a.rowAt(row), 0..) |v, i| {
                        sum += v * bT.m[col*bT.stride + i];
                    }
                    const v = res.get(row, col) + @reduce(.Add, sum);
                    res.set(row, col, v);
                }
            }
        }

        pub fn print(a: @This(), writer: anytype) !void {
            for (0..a.rows) |row| {
                for (a.rowAt(row)) |v| {
                    inline for (comptime 0..length(Vec)) |k| {
                        try writer.print("{} ", .{v[k]});
                    }
                }
                try writer.writeByte('\n');
            }
        }
    };
}

fn length(comptime V: type) comptime_int { return @typeInfo(V).Vector.len; }

fn readInt(comptime T: type, tokens: *mem.TokenIterator(u8)) !T {
    const token = tokens.next() orelse return error.Eof;
    return try std.fmt.parseInt(T, token, 10);
}

fn readMatrix(allocator: mem.Allocator, comptime T: type, rows: u16, cols: u16, tokens: *mem.TokenIterator(u8)) !Matrix(T) {
    const res = try Matrix(T).initUndefined(allocator, rows, cols);
    errdefer res.deinit(allocator);
    for (res.items()) |*v| {
        inline for (comptime 0..length(Matrix(T).Vec)) |k| {
            v.*[k] = try readInt(T, tokens);
        }
    }
    return res;
}

fn toSeconds(ns: u64) f64 { return 1e-3 * @intToFloat(f64, ns/1000_000); }

pub fn main() !void {
    var timer = try std.time.Timer.start();

    var arenaState = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arenaState.deinit();
    const arena = arenaState.allocator();

    const input = try std.io.getStdIn().reader().readAllAlloc(arena, 256*1024*1024);
    var tokens = std.mem.tokenize(u8, input, "\r\t\n ");
    const n = try readInt(u16, &tokens);
    const a = try readMatrix(arena, i32, n, n, &tokens);
    const b = try readMatrix(arena, i32, n, n, &tokens);
    const res = try Matrix(i32).initUndefined(arena, a.rows, b.cols);
    const temp = try arena.alloc(Matrix(i32).Vec, res.items().len);

    const stdout = std.io.getStdOut().writer();
    try stdout.print("read: {d:.3}s\n", .{toSeconds(timer.lap())});

    res.setAll(0).addProductRhsTransposed(a, b.copyTransposed(temp));

    try stdout.print("mul SSE: {d:.3}s\n", .{toSeconds(timer.lap())});

    res.setAll(0).addProduct(a, b, temp);

    try stdout.print("mul Strassen: {d:.3}s\n", .{toSeconds(timer.lap())});

    if (false) {
        var bufferedStdOut = std.io.bufferedWriter(stdout);
        try res.print(bufferedStdOut.writer());
        try bufferedStdOut.flush();
        try stdout.print("write: {d:.3}s\n", .{toSeconds(timer.lap())});
    }
}
