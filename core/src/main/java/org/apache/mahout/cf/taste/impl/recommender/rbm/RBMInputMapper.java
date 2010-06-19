/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.cf.taste.impl.recommender.rbm;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;

public class RBMInputMapper extends Mapper<LongWritable, Text, IntWritable, DistributedRowMatrix.MatrixEntryWritable> {

  @Override
  protected void map(LongWritable key, Text value, Context context) 
            throws IOException, InterruptedException {
    
    String [] entry = value.toString().split(",");
      
    //User is the key for the Reducer
    DistributedRowMatrix.MatrixEntryWritable record = 
      new DistributedRowMatrix.MatrixEntryWritable();
    IntWritable row = new IntWritable(Integer.valueOf(entry[0]));
    record.setRow(-1);
    record.setCol(Integer.valueOf(entry[1]));
    record.setVal(Double.valueOf(entry[2]));
    context.write(row, record);
  }
}
