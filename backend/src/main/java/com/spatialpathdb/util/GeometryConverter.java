package com.spatialpathdb.util;

import org.locationtech.jts.geom.Geometry;
import org.locationtech.jts.geom.GeometryFactory;
import org.locationtech.jts.geom.PrecisionModel;
import org.locationtech.jts.io.ParseException;
import org.locationtech.jts.io.WKTReader;
import org.locationtech.jts.io.WKTWriter;
import org.springframework.stereotype.Component;

@Component
public class GeometryConverter {

    private final GeometryFactory geometryFactory;
    private final WKTReader wktReader;
    private final WKTWriter wktWriter;

    public GeometryConverter() {
        // SRID 0 for pixel coordinates (no geographic projection)
        this.geometryFactory = new GeometryFactory(new PrecisionModel(), 0);
        this.wktReader = new WKTReader(geometryFactory);
        this.wktWriter = new WKTWriter();
    }

    public Geometry fromWkt(String wkt) {
        if (wkt == null || wkt.isEmpty()) {
            return null;
        }
        try {
            return wktReader.read(wkt);
        } catch (ParseException e) {
            throw new IllegalArgumentException("Invalid WKT: " + wkt, e);
        }
    }

    public String toWkt(Geometry geometry) {
        if (geometry == null) {
            return null;
        }
        return wktWriter.write(geometry);
    }

    public GeometryFactory getGeometryFactory() {
        return geometryFactory;
    }
}
