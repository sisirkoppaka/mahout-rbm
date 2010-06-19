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

package org.apache.mahout.cf.taste.example.netflix;

import java.io.IOException;
import java.io.StringReader;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Pattern;

import org.apache.commons.lang.StringEscapeUtils;
import org.apache.hadoop.io.DefaultStringifier;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.VLongWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.util.GenericsUtil;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Maps over Netflix CSV format
 * 
 */
public class NetflixDatasetCreatorMapper extends MapReduceBase implements
    Mapper<LongWritable,Text,VLongWritable,VLongWritable> {
  
  private static final Logger log = LoggerFactory.getLogger(NetflixDatasetCreatorMapper.class);
    
  @Override
  public void map(LongWritable key, Text value,
                  OutputCollector<VLongWritable,VLongWritable> output, Reporter reporter) throws IOException {
  
    //This DataSetCreatorMapper and DataSetCreatorReducer may not be required. Remove stubs when done with rest of algo.
  }
  
  @Override
  public void configure(JobConf job) {
    log.info("Configure: Job");
  }
}