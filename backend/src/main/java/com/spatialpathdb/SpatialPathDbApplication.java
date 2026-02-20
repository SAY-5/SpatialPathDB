package com.spatialpathdb;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableAsync;

@SpringBootApplication
@EnableAsync
public class SpatialPathDbApplication {

    public static void main(String[] args) {
        SpringApplication.run(SpatialPathDbApplication.class, args);
    }
}
