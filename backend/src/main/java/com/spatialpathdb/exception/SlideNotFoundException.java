package com.spatialpathdb.exception;

import java.util.UUID;

public class SlideNotFoundException extends RuntimeException {

    public SlideNotFoundException(UUID slideId) {
        super("Slide not found: " + slideId);
    }

    public SlideNotFoundException(String message) {
        super(message);
    }
}
