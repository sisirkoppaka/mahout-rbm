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

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.net.URI;
import java.text.DecimalFormat;
import java.text.NumberFormat;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.compress.BZip2Codec;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.FileLineIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Split the Netflix CSV dataset into chunks of size as specified on the command line
 * 
 */
public final class NetflixCSVSplitter {
  
  private static final Logger log = LoggerFactory.getLogger(NetflixCSVSplitter.class);
  
  private NetflixCSVSplitter() { }
  
  public static void main(String[] args) throws IOException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option dumpFileOpt = obuilder.withLongName("dumpFile").withRequired(true).withArgument(
      abuilder.withName("netflixDumpFile").withMinimum(1).withMaximum(1).create()).withDescription(
      "The path to the Netflix dataset CSV dump file (.bz2 or uncompressed)").withShortName("d").create();
    
    Option outputDirOpt = obuilder.withLongName("outputDir").withRequired(true).withArgument(
      abuilder.withName("outputDir").withMinimum(1).withMaximum(1).create()).withDescription(
      "The output directory to place the splits in:\n"
          + "local files:\n\t/var/data/netflix-csv-chunks or\n\tfile:///var/data/netflix-csv-chunks\n"
          + "Hadoop DFS:\n\thdfs://netflix-csv-chunks\n"
          + "AWS S3 (blocks):\n\ts3://bucket-name/netflix-csv-chunks\n"
          + "AWS S3 (native files):\n\ts3n://bucket-name/netflix-csv-chunks\n")

    .withShortName("o").create();
    
    Option s3IdOpt = obuilder.withLongName("s3ID").withRequired(false).withArgument(
      abuilder.withName("s3Id").withMinimum(1).withMaximum(1).create()).withDescription("Amazon S3 ID key")
        .withShortName("i").create();
    Option s3SecretOpt = obuilder.withLongName("s3Secret").withRequired(false).withArgument(
      abuilder.withName("s3Secret").withMinimum(1).withMaximum(1).create()).withDescription(
      "Amazon S3 secret key").withShortName("s").create();
    
    Option chunkSizeOpt = obuilder.withLongName("chunkSize").withRequired(true).withArgument(
      abuilder.withName("chunkSize").withMinimum(1).withMaximum(1).create()).withDescription(
      "Chunk size, in megabytes").withShortName("c").create();
    Option numChunksOpt = obuilder
        .withLongName("numChunks")
        .withRequired(false)
        .withArgument(abuilder.withName("numChunks").withMinimum(1).withMaximum(1).create())
        .withDescription(
          "The maximum number of chunks to create.  If specified, program will only create a subset of the chunks")
        .withShortName("n").create();
    Group group = gbuilder.withName("Options").withOption(dumpFileOpt).withOption(outputDirOpt).withOption(
      chunkSizeOpt).withOption(numChunksOpt).withOption(s3IdOpt).withOption(s3SecretOpt).create();
    
    Parser parser = new Parser();
    parser.setGroup(group);
    CommandLine cmdLine;
    try {
      cmdLine = parser.parse(args);
    } catch (OptionException e) {
      log.error("Error while parsing options", e);
      CommandLineUtil.printHelp(group);
      return;
    }
    
    Configuration conf = new Configuration();
    String dumpFilePath = (String) cmdLine.getValue(dumpFileOpt);
    String outputDirPath = (String) cmdLine.getValue(outputDirOpt);
    
    if (cmdLine.hasOption(s3IdOpt)) {
      String id = (String) cmdLine.getValue(s3IdOpt);
      conf.set("fs.s3n.awsAccessKeyId", id);
      conf.set("fs.s3.awsAccessKeyId", id);
    }
    if (cmdLine.hasOption(s3SecretOpt)) {
      String secret = (String) cmdLine.getValue(s3SecretOpt);
      conf.set("fs.s3n.awsSecretAccessKey", secret);
      conf.set("fs.s3.awsSecretAccessKey", secret);
    }
    conf.set("fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem");
    FileSystem fs = FileSystem.get(URI.create(outputDirPath), conf);
    
    int chunkSize = 1024 * 1024 * Integer.parseInt((String) cmdLine.getValue(chunkSizeOpt));
    
    int numChunks = Integer.MAX_VALUE;
    if (cmdLine.hasOption(numChunksOpt)) {
      numChunks = Integer.parseInt((String) cmdLine.getValue(numChunksOpt));
    }
    
    StringBuilder chunkContent = new StringBuilder();
    NumberFormat decimalFormatter = new DecimalFormat("0000");
    File dumpFile = new File(dumpFilePath);
    FileLineIterator it;
    if (dumpFilePath.endsWith(".bz2")) {
      CompressionCodec codec = new BZip2Codec();
      it = new FileLineIterator(codec.createInputStream(new FileInputStream(dumpFile)));
    } else {
      it = new FileLineIterator(dumpFile);
    }
    int filenumber = 0;
    while (it.hasNext()) {
      String thisLine = it.next();  
      if (chunkContent.length() > chunkSize) {
        filenumber++;
        String filename = outputDirPath + "/chunk-" + decimalFormatter.format(filenumber) + ".csv";
        BufferedWriter chunkWriter = new BufferedWriter(new OutputStreamWriter(fs
              .create(new Path(filename)), "UTF-8"));
          
        chunkWriter.write(chunkContent.toString(), 0, chunkContent.length());
        chunkWriter.close();
        if (filenumber >= numChunks) {
          break;
        }
        chunkContent = new StringBuilder();
      }
    }
  }
}
