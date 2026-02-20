package com.spatialpathdb.config;

import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Contact;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.info.License;
import io.swagger.v3.oas.models.servers.Server;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;

@Configuration
public class SwaggerConfig {

    @Bean
    public OpenAPI customOpenAPI() {
        return new OpenAPI()
            .info(new Info()
                .title("SpatialPathDB API")
                .version("1.0.0")
                .description("""
                    REST API for spatial pathology data management and analytics.

                    Features:
                    - Slide metadata management
                    - High-performance spatial queries (bbox, KNN, density)
                    - Asynchronous analysis job processing
                    - Benchmarking and performance monitoring
                    """)
                .contact(new Contact()
                    .name("SpatialPathDB Team")
                    .email("support@spatialpathdb.io"))
                .license(new License()
                    .name("MIT License")
                    .url("https://opensource.org/licenses/MIT")))
            .servers(List.of(
                new Server()
                    .url("http://localhost:8080")
                    .description("Local development server"),
                new Server()
                    .url("http://backend:8080")
                    .description("Docker container")
            ));
    }
}
