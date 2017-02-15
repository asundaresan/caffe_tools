#!/usr/bin/env python 

import sqlite3
import cv2
import os
import argparse
import numpy as np

def get_face_box( sqlite_file, face_folder = "", not_face_folder = "", nimages = 0, verbosity = 0 ):
  counter = 1

  # Open the sqlite database
  conn = sqlite3.connect(sqlite_file)
  c = conn.cursor()

  select_string = "faceimages.filepath, facerect.x, facerect.y, facerect.w, facerect.h, faces.face_id"
  from_string = "faceimages, faces, facepose, facerect"
  where_string = "faces.face_id = facepose.face_id and faces.file_id = faceimages.file_id and faces.face_id = facerect.face_id"
  query_string = "SELECT " + select_string + " FROM " + from_string + " WHERE " + where_string

  first_time = True

  # It iterates through the rows returned from the query
  for (counter, row ) in enumerate( c.execute(query_string) ):
    image_file = str( row[0] )
    face_x = max( 0, row[1] )
    face_y = max( 0, row[2] )
    face_w = row[3]
    face_h = row[4]
    face_id = row[5]

    # Creating the full path names for input and output
    input_path = "%s/flickr/%s" % ( os.path.dirname( sqlite_file ), image_file )

    #If the file exist then open it       
    if os.path.isfile( input_path ):
        image = cv2.imread( input_path, cv2.IMREAD_COLOR ) #load the colour version

        #Image dimensions
        image_h, image_w, nchannels = image.shape

        face_w = min( face_w, image_w - face_x )
        face_h = min( face_h, image_h - face_y )

        if face_folder != "":
          if first_time:
            print( "Will write faces to: %s" % face_folder )
            if not os.path.isdir( face_folder ):
              os.makedirs( face_folder )
            first_time = False
          output_path = "%s/%09d.jpg" % ( face_folder, counter )
          image_cropped = np.copy(image[face_y:face_y+face_h, face_x:face_x+face_w])
          cv2.imwrite(output_path, image_cropped)

        if verbosity > 1:
          pt1 = ( face_x, face_y )
          pt2 = ( face_x + face_w, face_y + face_h )
          cv2.rectangle( image, pt1, pt2, (0,255,0) )
          cv2.imshow( "Face", image )
          wait_time = 0 if verbosity > 2 else 500
          ret = cv2.waitKey( wait_time )
          if ret == ord( "q" ):
            break
        if nimages > 0 and counter > nimages:
          break
    else:
      raise ValueError('Error: I cannot find the file specified: ' + str(input_path))

  #Once finished the iteration it closes the database
  c.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument( "db_file", help = "SQLite database file" )
  parser.add_argument( "--face_folder", "-f", default = "", help = "Output face folder" )
  parser.add_argument( "--not_face_folder", "-n", default = "", help = "Output not-face folder" )
  parser.add_argument( "--verbosity", "-v", action="count", default = 0, help="Verbosity level" )
  args = parser.parse_args()

  db_file = args.db_file
  face_folder = "%s/faces" % os.path.dirname( db_file ) if args.face_folder == "" else args.face_folder 
  not_faces_folder = "not_faces" if args.not_face_folder == "" else args.not_face_folder 
  get_face_box( db_file, face_folder = face_folder )


