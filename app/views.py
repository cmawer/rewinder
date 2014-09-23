import os
from flask import render_template, jsonify, request
import MySQLdb
from app import app
import folium
import rewinder as rewind
import grasp
import json
global start
import numpy as np
start=np.array([0,0])


#db = MySQLdb.connect(user="root", host="localhost", passwd="encycsql",port=3306, db="world_inn")

@app.route("/")
def hello():
    print "hi!"
    return render_template('indext.html') 

@app.route("/broute",methods=['GET'])
def routeit():
    user_address = request.args.get('address')
    global desired_distance
    desired_distance = float(request.args.get('miles'))
    global start
    start=rewind.geocode(user_address)
    
    
    elevation_change_indicator=request.args.get('elevation')
    if elevation_change_indicator:
        elevation_change_indicator=1
    else:
        elevation_change_indicator=0
    
    global distances_sorted, paths_sorted, numbers_seen_sorted, possible_locations
    opt_route_locations, distance, direction_coordinates, narratives, added_indicator, paths_sorted, distances_sorted, numbers_seen_sorted, possible_locations, stops, mean_elevation_change, max_elevation_change,elevations,distances_el=rewind.main(start,desired_distance,elevation_change_indicator)
    global start
    star=[start[0][0],start[0][1]]
    m = folium.Map(location=star,zoom_start=13,tiles='Mapbox',API_key='cmm34.jdomfk86')
    m.line(direction_coordinates[0], line_color='#2DD0AF', line_weight=5)
    j=-1
    n=-1-added_indicator
    for coor in opt_route_locations[:n]:
        j=j+1
        m.simple_marker(coor,popup=stops[j])
    j=-1
    global photo_locations, photo_urls
    photo_locations,photo_urls=rewind.getPhotos(start)
    for coor in photo_locations:
        j=j+1
        if j<1800:
            ima='<a><img src=http://static.panoramio.com/photos/large/'+ photo_urls[j].split('/')[-1]+'.jpg height="150px" width="200px"></a>'
            m.circle_marker(coor, radius=50,popup=ima, line_color='#674ea7',fill_color='#674ea7', fill_opacity=0.2)
    m.create_map(path='app/templates/osm.html')

    return render_template('route.html',dist=distance,numseen=len(opt_route_locations[:n])-1,narrative=narratives[0],stops=stops,mean_elev=int(np.round(mean_elevation_change)),max_elev=int(np.round(max_elevation_change)))

@app.route("/makemap")
def makemap():
    return render_template('osm.html')

@app.route("/newroute")
def newroute():
    global paths_sorted, distances_sorted, numbers_seen_sorted, possible_locations, start, desired_distance
    new_route, distance, direction_coordinates, narratives, added_indicator, stops,elevations,distances_el,mean_elevation_change,max_elevation_change=rewind.newRoute(paths_sorted, distances_sorted, numbers_seen_sorted, possible_locations,start,desired_distance)
    global start
    star=[start[0][0],start[0][1]]
    m = folium.Map(location=star,zoom_start=13,tiles='Mapbox',API_key='cmm34.jdomfk86')
    m.line(direction_coordinates[0], line_color='#2DD0AF', line_weight=5)
    j=-1
    n=-1-added_indicator
    for coor in new_route[:n]:
        j=j+1
        m.simple_marker(coor,popup=stops[j])
    j=-1
    global photo_locations, photo_urls
    photo_locations,photo_urls=rewind.getPhotos(start)
    for coor in photo_locations:
        j=j+1
        if j<1800:
            ima='<a><img src=http://static.panoramio.com/photos/large/'+ photo_urls[j].split('/')[-1]+'.jpg height="150px" width="200px"></a>'
            m.circle_marker(coor, radius=50,popup=ima, line_color='#674ea7',fill_color='#674ea7', fill_opacity=0.2)
    m.create_map(path='app/templates/osm.html')

    return render_template('route.html',dist=distance,numseen=len(new_route[:n])-1,narrative=narratives[0],stops=stops,mean_elev=int(np.round(mean_elevation_change)),max_elev=int(np.round(max_elevation_change)))
    
@app.route("/slides")
def slide():
    return render_template('slides.html')