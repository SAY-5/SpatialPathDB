package com.spatialpathdb.util;

import lombok.extern.slf4j.Slf4j;

import java.util.function.Supplier;

@Slf4j
public class QueryTimer {

    public static <T> TimedResult<T> time(Supplier<T> operation) {
        long start = System.nanoTime();
        T result = operation.get();
        long elapsed = (System.nanoTime() - start) / 1_000_000;
        return new TimedResult<>(result, elapsed);
    }

    public static <T> TimedResult<T> timeAndLog(String operationName, Supplier<T> operation) {
        long start = System.nanoTime();
        T result = operation.get();
        long elapsed = (System.nanoTime() - start) / 1_000_000;

        log.debug("{} completed in {}ms", operationName, elapsed);

        return new TimedResult<>(result, elapsed);
    }

    public record TimedResult<T>(T result, long elapsedMs) {}
}
