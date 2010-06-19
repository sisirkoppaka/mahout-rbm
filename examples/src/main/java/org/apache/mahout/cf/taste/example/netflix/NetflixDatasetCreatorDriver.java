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

import java.io.File;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DefaultStringifier;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.util.GenericsUtil;
import org.apache.lucene.analysis.Analyzer;
import org.apache.mahout.analysis.WikipediaAnalyzer;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.FileLineIterable;
import org.apache.mahout.common.HadoopUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Create and run the Netflix Dataset Creator.
 */
public final class NetflixDatasetCreatorDriver {
  private static final Logger log = LoggerFactory.getLogger(NetflixDatasetCreatorDriver.class);
  
  private NetflixDatasetCreatorDriver() { }
  
  public static void main(String[] args) throws IOException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    
  }
  

  public static void runJob() throws IOException {
    JobClient client = new JobClient();
    JobConf conf = new JobConf(NetflixDatasetCreatorDriver.class);
    if (NetflixDatasetCreatorDriver.log.isInfoEnabled()) {
    }
   //fill in
    JobClient.runJob(conf);
  }
}
