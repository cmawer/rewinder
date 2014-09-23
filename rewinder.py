import json
from pprint import pprint
import requests
import numpy as np
import pandas as pd
import time 
from app import app
from flask import render_template, request
import os
import sys
import MySQLdb as sql
from copy import deepcopy

direct_to_walk_frac=0.4
R=3963.1676
url_optroute='http://open.mapquestapi.com/directions/v2/optimizedroute?key=Fmjtd%7Cluur2h6y2u%2Ca2%3Do5-9wa2dz&routeType=pedestrian'
url_route='http://open.mapquestapi.com/directions/v2/route?key=Fmjtd%7Cluur2h6y2u%2Ca2%3Do5-9wa2dz&routeType=pedestrian'
url_geocode='http://open.mapquestapi.com/geocoding/v1/address?key=Fmjtd%7Cluur2h6y2u%2Ca2%3Do5-9wa2dz&location='
url_matrix='http://open.mapquestapi.com/directions/v2/routematrix?key=Fmjtd%7Cluur2h6y2u%2Ca2%3Do5-9wa2dz&routeType=pedestrian'
url_elevation='http://open.mapquestapi.com/elevation/v1/profile?key=Fmjtd%7Cluur2h6y2u%2Ca2%3Do5-9wa2dz&routeType=pedestrian&shapeFormat=raw&unit=f&latLngCollection='

# Following functions manipulate latitude,longitude coordinates 

def newLatLong(pt_in,r,theta):
    #Finds the lat long of a point distance r from original lat, long in direction theta
    #where theta starts north (theta=0) and goes clockwise - assumes distance extends 
    # around less than one quarter of the circumference of the earth in longitude
    lat_in=np.radians(pt_in[:,0])
    long_in=np.radians(pt_in[:,1])
    lat_out=np.arcsin(np.sin(lat_in)*np.cos(r/R)+np.cos(lat_in)*np.sin(r/R)*np.cos(theta))
    long_out=long_in-np.arctan2(np.sin(theta)*np.sin(r/R)*np.cos(lat_in),np.cos(r/R)-np.sin(lat_in)*np.sin(lat_out))
    lat_out=np.degrees(lat_out)
    long_out=np.degrees(long_out)
    pt_out=np.reshape(np.array([lat_out,long_out]),(1,2))
    return pt_out

def makeBox(pt_in,D):
    # Given point lat,long finds the [min lat, max lat, min long, max long] that defines a 
    # square with side length d
    theta=[0, np.pi/2, np.pi, 3*np.pi/2]
    lat_box=[]
    long_box=[]
    r=np.true_divide(D,2)
    for t in theta:
        pt_out=newLatLong(pt_in,r,t)
        lat_box.append(pt_out[:,0])
        long_box.append(pt_out[:,1])
    latlong_box=np.array([np.min(lat_box), np.min(long_box), np.max(lat_box), np.max(long_box)])
    return latlong_box

def getDist(pt1,pt2):
    # Finds distance between pt1 and pt2 where pt = [lat, long]
    #pt1 = [lat, long] 
    pt1=np.radians(pt1)
    pt2=np.radians(pt2)
    dpt=pt2-pt1
    a=np.square(np.sin(np.true_divide(dpt[:,0],2)))+np.cos(pt1[:,0])*np.cos(pt2[:,0])*np.square(np.sin(np.true_divide(dpt[:,1],2)))
    c=np.multiply(2,np.arctan2(np.sqrt(a),np.sqrt(1-a)))
    dist_out=np.array(np.multiply(R,c))
    return dist_out

def inCircle(pt_in,pts,r):
    # Returns only the points that are within radius r from pt1 as well as the
    # corresponding indices
    pt1=np.tile(np.array(pt_in),(len(pts[:,0]),1))
    dists_out=getDist(pt1,pts)
    index_circle=dists_out<r
    pts_in_circle=pts[index_circle] 
    return pts_in_circle, index_circle

def getLocDict(pts_in):
    location_dict=[]
    for lat,long in pts_in:
        dict={'latLng':{'lat':lat,'lng':long}}
        location_dict.append(dict)
    return location_dict

#Following functions use Mapquest APIs to get route information 

def getOptRouteMQ(pts_in): 
    #Given input of points (up to 50), finds optimal route through them and outputs distance
    location_dict=getLocDict(pts_in)
    body={'locations':location_dict}
    data_json = json.dumps(body)
    api_response = requests.get(url_optroute, data=data_json)
    distance_traveled=api_response.json()['route']['distance']
    sequence_of_stops=np.array(api_response.json()['route']['locationSequence'])
    return distance_traveled,sequence_of_stops,api_response

def getRegRouteMQ(pts_in): 
    #Given input of points (up to 50), finds route through them in order given and outputs distance
    location_dict=getLocDict(pts_in)
    body={'locations':location_dict}
    data_json = json.dumps(body)
    api_response = requests.get(url_route, data=data_json)
    distance_traveled=api_response.json()['route']['distance']
    sequence_of_stops=np.array(api_response.json()['route']['locationSequence'])
    return distance_traveled,sequence_of_stops,api_response

   
def geocode(address_in):
    #Given string input of address, outputs lat/long
    url=url_geocode+address_in
    api_response=requests.get(url)
    try:
        lat_in=api_response.json()['results'][0]['locations'][0]['latLng']['lat']
        long_in=api_response.json()['results'][0]['locations'][0]['latLng']['lng']
        start=np.reshape(np.array([lat_in,long_in]),(1,2))
    except:
        start=pt=np.reshape(np.array([0,0]),(1,2))
    return start

def getMatrixMQ(pts_in):
    #Can only give 25 places
    url=url_matrix+'&alltoAll=true'
    location_dict=getLocDict(pts_in[0:25])
    body={'locations':location_dict,'options': {'allToAll': 'true'}}
    data_json = json.dumps(body)
    api_response = requests.get(url, data=data_json)
    matrix=np.array(api_response.json()['distance'])
    return matrix

def getElevationMQ(pts_in):
    url=url_elevation
    for lat_in,long_in in pts_in:
        url=url+str(lat_in)+','+str(long_in)+','
    api_response = requests.get(url[0:-1])
    #elevations=np.array(api_response.json()['distance'])
    elevations=np.array([])
    distances_el=np.array([])
    for elev in api_response.json()['elevationProfile']:
        elevations=np.append(elevations,elev['height'])
        distances_el=np.append(distances_el,elev['distance'])
    return np.array(elevations),np.array(distances_el)


def getMatrixOne(pts_in,start):
    distances_to_start=np.array([])
    for j in range(0,int(np.ceil(np.true_divide(len(pts_in),100)))):
        location_dict=getLocDict(np.concatenate((start,pts_in[j*99:(j+1)*100-1]),axis=0))
        body={'locations':location_dict,'options': {'allToAll': 'false'}}
        data_json = json.dumps(body)
        api_response = requests.get(url_matrix, data=data_json)
        if j==0:
            distances_to_start=np.concatenate((distances_to_start,api_response.json()['distance']))
        else:
            distances_to_start=np.concatenate((distances_to_start,api_response.json()['distance'][1:]))   
    return distances_to_start
    
def getMatrixOneDirect(pts_in,start):
    pt1=np.tile(start,(len(pts_in),1))
    distances_to_start_direct=np.concatenate((np.array([0]),getDist(pt1,pts_in)))
    distances_to_start_direct=np.true_divide(distances_to_start_direct,direct_to_walk_frac)
    return distances_to_start_direct
    
def getMatrixDirect(pts_in):
    j=-1
    a=0
    matrix=np.zeros((len(pts_in),len(pts_in)))
    for x in pts_in:
        j=j+1
        a=a+1
        pt1=np.tile(np.array(x),(len(pts_in[a:,0]),1))
        dists=getDist(pt1,pts_in[a:])
        matrix[j,a:]=dists
        matrix[a:,j]=dists
    matrix=np.true_divide(matrix,direct_to_walk_frac)
    return matrix
    
#The following function pulls locations from MySQL

def getPossibleSites(latlong_box):
    conn = sql.connect(host='localhost',port=int(3306),user='root',passwd='encycsql',db='insightplaces')
    command='SELECT Place,Latitude,Longitude FROM Places WHERE Latitude > '+str(latlong_box[0])+' AND Latitude< '+str(latlong_box[2])+' AND Longitude> '+str(latlong_box[1])+' AND Longitude< '+str(latlong_box[3])
    places=[]
    lats=[]
    longs=[]
    with conn: 
        cur = conn.cursor()
        cur.execute(command)
        rows = cur.fetchall()
        for row in rows:
            lats.append(float(row[1]))
            longs.append(float(row[2]))
    possible_locations=np.array(zip(lats,longs))
    del lats,longs
    return possible_locations

def getPhotos(start):
    photo_latlong_box=makeBox(start,10)
    #Get nearby historical places
    conn = sql.connect(host='localhost',port=int(3306),user='root',passwd='encycsql',db='insightplaces')
    command='SELECT Addresses,Latitude,Longitude FROM Photos WHERE Latitude > '+str(photo_latlong_box[0])+' AND Latitude< '+str(photo_latlong_box[2])+' AND Longitude> '+str(photo_latlong_box[1])+' AND Longitude< '+str(photo_latlong_box[3])
    photo_urls=[]
    photo_lats=[]
    photo_longs=[]
    with conn: 
        cur = conn.cursor()
        cur.execute(command)
        rows = cur.fetchall()
        for row in rows:
            photo_urls.append(row[0])
            photo_lats.append(float(row[1]))
            photo_longs.append(float(row[2]))
    photo_locations=zip(photo_lats,photo_longs)
    return photo_locations, photo_urls 

# The following functions N routes for N sites 

def removeDistantSites(possible_locations,start,D):
    if len(possible_locations)>24:
        distances_to_start=getMatrixOneDirect(possible_locations,start)
    else:
        distances_to_start=getMatrixOne(possible_locations,start) #Includes home to home at 0
    possible_locations=np.array(zip(np.extract(distances_to_start[1:]<np.true_divide(D,2),possible_locations[:,0]),np.extract(distances_to_start[1:]<np.true_divide(D,2),possible_locations[:,1])))
    distances_to_home=np.extract(distances_to_start<np.true_divide(D,2),distances_to_start)
    return possible_locations, distances_to_home

def getSingleRoute(matrix,D,ind_start):
    distance_left=1.1*D
    distance_to_travel=matrix[0,ind_start] #distance from current point to first stop
    distance_home=matrix[ind_start,0] #distance from first stop home 
    distance_home_prev=0
    j0=0
    j=ind_start
    x=[]
    while distance_home<distance_left-distance_to_travel:
        x.append(j) # Can make it home so can go to this point 
        distance_left=distance_left-distance_to_travel #Move to point j
        matrix[j,0]=1000 # Don't want to go home yet
        matrix[:,j]=1000 # Don't want to pick 0 for this location or let it be chosen again
        j0=j 
        j=np.argmin(matrix[j,:]) #Find closest point to current
        distance_to_travel=matrix[j0,j] #Find distance to travel from previous point to this point
        distance_home_prev=distance_home
        distance_home=matrix[j,0] #Distance home from next point 
    distance_left=distance_left-distance_home_prev
    number_seen=len(x)
    distance_traveled=1.1*D-distance_left
    locations_seen=np.array(x)
    del x 
    return distance_traveled, number_seen, locations_seen

def getAllRoutes(matrix,D):
    paths=[]
    numbers_seen=[]
    path_distances=[]
    for ind_start in range(1,len(matrix[0,:])-1):
        distance_traveled, number_seen, locations_seen=getSingleRoute(matrix,D,ind_start)
        numbers_seen.append(number_seen)
        paths.append(locations_seen)
        path_distances.append(distance_traveled)
    numbers_seen=np.array(numbers_seen)
    paths=np.array(paths)
    path_distances=np.array(path_distances)
    return numbers_seen,paths,path_distances

def findOptRoute(possible_locations,D,start,elevation_change_indicator):
    possible_locations,distances_to_home=removeDistantSites(possible_locations,start,D)
    if len(possible_locations)>24:
        # distances_to_start_direct=getMatrixOneDirect(possible_locations,start)
#         direct_to_walk_frac=np.min(np.true_divide(distances_to_start_direct,distances_to_home[1:]))
        # print direct_to_walk_frac
#         print np.mean(np.true_divide(distances_to_start_direct,distances_to_home[1:]))
        matrix=getMatrixDirect(np.concatenate((start,possible_locations)))
    else:
        matrix=getMatrixMQ(np.concatenate((start,possible_locations)))
        
    numbers_seen,paths,path_distances=getAllRoutes(matrix,D)
    
    # Pick best route and rank the rest 
    opt_routes_ind=np.argwhere(numbers_seen == np.amax(numbers_seen)).flatten().tolist()
    opt_route_max_dist_ind=opt_routes_ind[np.argmax(np.array(path_distances)[opt_routes_ind])]
    opt_path=paths[opt_route_max_dist_ind]
    paths=np.extract(numbers_seen>0,paths)
    path_distances=np.extract(numbers_seen>0,path_distances)
    numbers_seen=np.extract(numbers_seen>0,numbers_seen)
    paths_sorted_ind=np.argsort(numbers_seen)[::-1]
    numbers_seen_sorted=numbers_seen[paths_sorted_ind]
    distances_sorted=path_distances[paths_sorted_ind]
    paths_sorted=paths[paths_sorted_ind]
    percent_diff_distance=np.true_divide(np.abs(distances_sorted-D),D)
    
    
    
    
    
    #Adjust path ranking if flat elevation desired
    if elevation_change_indicator==1:
        #Don't want to add a point too far because will change elevation when adding a point 
        numbers_seen_sorted=np.extract(percent_diff_distance<0.15,numbers_seen_sorted)
        paths_sorted=np.extract(percent_diff_distance<0.15,paths_sorted)
        distances_sorted=np.extract(percent_diff_distance<0.15,distances_sorted)
        
        elevation_rank,mean_elevation_change,max_elevation_change=rankElevations(possible_locations,paths_sorted,start)
        paths_sorted=paths_sorted[elevation_rank]
        opt_path=paths_sorted[0]
    
    # Get the locations along the route 
    opt_route_locations=np.array(start)
    for j in opt_path:
        x=np.reshape(possible_locations[j-1],(1,2)) #Because j was calculated from array with starting point at 0
        opt_route_locations=np.concatenate((opt_route_locations,x))
    opt_route_locations=np.concatenate((opt_route_locations,start)) 
    
    direction_coordinates,narratives,distance=getFinalRoute(opt_route_locations)
    
    if distance<0.9*D:
        opt_route_locations,direction_coordinates, narratives, distance,added_indicator=addPoint(opt_route_locations,D,distance)
    else:
        added_indicator=0
    elevations,distances_el=getElevationMQ(direction_coordinates[0])
    mean_elevation_change=np.mean(np.true_divide(np.abs(np.diff(elevations)),distances_el[1:]))
    max_elevation_change=np.max(np.true_divide(np.abs(np.diff(elevations)),distances_el[1:]))
    
    return opt_route_locations, distance, direction_coordinates, narratives, added_indicator, paths_sorted, distances_sorted, numbers_seen_sorted, possible_locations,mean_elevation_change,max_elevation_change,elevations,distances_el
        
def getFinalRoute(opt_route_locations):
    K=int(np.ceil(np.true_divide(len(opt_route_locations),25)))
    distance=0
    direction_coordinates=[]
    narratives=[]
    for j in range(0,K):
        distance_traveled,sequence_of_stops,api_response=getRegRouteMQ(opt_route_locations[j*25:(j+1)*25+1])
        c,na=getDirections(api_response)
        direction_coordinates.append(c)
        narratives.append(list(na))
        distance=distance_traveled+distance
    return direction_coordinates,narratives,distance

def getDirections(api_response):
    directions=[]
    coordinates=[]
    print api_response.json()
    for mans in api_response.json()['route']['legs']:
                for sps in mans['maneuvers']:
                    lat=sps['startPoint']['lat']
                    long=sps['startPoint']['lng']
                    coordinates.append([lat,long])
                    narr=sps['narrative']
                    directions.append(narr)
    coordinates=np.array(coordinates)
    return coordinates,directions

    
def getStops(opt_route_locations,added_indicator):
    conn = sql.connect(host='localhost',port=int(3306),user='root',passwd='encycsql',db='insightplaces')
    n=-1-added_indicator
    stops=['Home']
    for stop in opt_route_locations[1:n]:
        command='SELECT Place FROM Places WHERE Latitude LIKE \''+str(stop[0])[0:7]+'%\' AND Longitude LIKE \''+str(stop[1])[0:9]+'%\''          
        with conn:
            cur = conn.cursor()
            cur.execute(command)
            row = cur.fetchall()
            if not row:
                stops.append('N/A')
            else:
                stops.append(row[0][0])
    return stops
    
def addPoint(opt_route_locations,D,distance):
    distance_left=D-distance
    print distance_left
    latlongbox1=makeBox(np.reshape(opt_route_locations[-1],(1,2)),np.multiply(distance_left,2))
    latlongbox2=makeBox(np.reshape(opt_route_locations[-2],(1,2)),np.multiply(distance_left,2))
    
    poss_lats=np.array([latlongbox1[0],latlongbox1[2],latlongbox2[0],latlongbox2[2]])
    lats_box=np.sort(poss_lats)[1:-1]
    poss_longs=np.array([latlongbox1[1],latlongbox1[3],latlongbox2[1],latlongbox2[3]])
    longs_box=np.sort(poss_longs)[1:-1]
    random_lats=np.random.random(23)*(np.max(lats_box)-np.min(lats_box))+np.min(lats_box)
    random_longs=np.random.random(23)*(np.max(longs_box)-np.min(longs_box))+np.min(longs_box)
    random_pts=zip(random_lats,random_longs)
    random_pts.insert(0,list(opt_route_locations[-2]))
    random_pts.append(list(opt_route_locations[-1]))
    matrix=getMatrixMQ(np.array(random_pts))
        
    possible_distances_home=matrix[0,:]+matrix[:,24]
    difference=np.abs(possible_distances_home-(distance_left+possible_distances_home[0])) #want to go distance_left+distance from -2 to -1
    ind_extra=np.argmin(difference[1:-1])
    extra_stop=np.reshape(np.array(random_pts[ind_extra+1]),(1,2))
    opt_route_locations_alt=np.concatenate((opt_route_locations[:-1],extra_stop,np.reshape(opt_route_locations[-1],(1,2))))
    
    direction_coordinates,narratives,distance_alt=getFinalRoute(opt_route_locations_alt)
    if np.abs(D-distance_alt)<np.abs(D-distance):
        opt_route_locations=opt_route_locations_alt
        distance=distance_alt
        added_indicator=1
    else:
        added_indicator=0
    
    return opt_route_locations,direction_coordinates, narratives, distance, added_indicator

def newRoute(paths_sorted, distances_sorted, numbers_seen_sorted, possible_locations,start,D):
    path=np.random.choice(paths_sorted)
    new_route=np.array(start)
    for j in path:
        x=np.reshape(possible_locations[j-1],(1,2)) #Because j was calculated from array with starting point at 0
        new_route=np.concatenate((new_route,x))
    new_route=np.concatenate((new_route,start))
    direction_coordinates,narratives,distance=getFinalRoute(new_route)
    
    if distance<0.9*D:
        new_route,direction_coordinates, narratives, distance,added_indicator=addPoint(new_route,D,distance)
    else:
        added_indicator=0
        
    stops=getStops(new_route,added_indicator)
    print direction_coordinates
    elevations,distances_el=getElevationMQ(direction_coordinates[0])
    mean_elevation_change=np.mean(np.true_divide(np.abs(np.diff(elevations)),distances_el[1:]))
    max_elevation_change=np.max(np.true_divide(np.abs(np.diff(elevations)),distances_el[1:]))
    return new_route, distance, direction_coordinates, narratives, added_indicator, stops,elevations,distances_el, mean_elevation_change,max_elevation_change
    
def rankElevations(possible_locations,paths_sorted,start):
    elevs,ds=getElevationMQ(possible_locations)
    start_elevation,ds=getElevationMQ(start)
    elevs=np.concatenate((start_elevation,elevs))
    elevation_change=[]
    for path in paths_sorted:
        path=np.concatenate((np.array([0]),path,np.array([0])))
        diffs=np.diff(elevs[path])
        elevation_change.append(diffs)
    mean_change=[]
    max_change=[]
    for path in elevation_change:
        mean_change.append(np.mean(np.abs(path)))
        max_change.append(np.max(np.abs(path)))
    elevation_rank=np.argsort(np.add(max_change,mean_change))
    mean_elevation_change=np.array(mean_change)[elevation_rank]
    max_elevation_change=np.array(max_change)[elevation_rank]
    return elevation_rank, mean_elevation_change,max_elevation_change

def main(start, D,elevation_change_indicator):
    latlong_box=makeBox(start,D)
    possible_locations=getPossibleSites(latlong_box)
    opt_route_locations, distance, direction_coordinates, narratives, added_indicator, paths_sorted, distances_sorted, numbers_seen_sorted, possible_locations,mean_elevation_change,max_elevation_change,elevations,distances_el=findOptRoute(possible_locations, D,start,elevation_change_indicator)
    print opt_route_locations
    stops=getStops(opt_route_locations,added_indicator)
    return opt_route_locations, distance, direction_coordinates, narratives, added_indicator, paths_sorted, distances_sorted, numbers_seen_sorted, possible_locations, stops, mean_elevation_change, max_elevation_change,elevations,distances_el
    
        
    
    
