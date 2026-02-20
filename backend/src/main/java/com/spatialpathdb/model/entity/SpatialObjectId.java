package com.spatialpathdb.model.entity;

import lombok.*;

import java.io.Serializable;
import java.util.UUID;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@EqualsAndHashCode
public class SpatialObjectId implements Serializable {

    private Long id;
    private UUID slideId;
}
