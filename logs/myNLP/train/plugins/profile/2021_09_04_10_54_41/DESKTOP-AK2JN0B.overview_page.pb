?+$	??????"H?*????y?&1?l?!??/?$??	!       "\
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails	??g????S?!?uq??A??ܵ?|??"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/n????a??+ei?AV-???"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails????Mbp???_vOf?A??_?LU?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsU???N@s?F%u?k?AǺ???V?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsn??t?y?&1?l?AǺ???V?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsa2U0*?s???H?}m?Aa2U0*?S?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?HP?x??????g?A-C??6j?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?s????a2U0*???AȘ?????"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsHP?sג??q????o?AV-???"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails	y?&1?l??????g?Aa2U0*?C?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails
HP?s?r?????Mbp?Aa2U0*?C?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails	?^)?p?y?&1?l?Aa2U0*?C?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??/?$???I+???A?n?????"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?~j?t?x????_vOn?AHP?s?b?"U
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails????o??????o??"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsˡE?????Έ?????Aŏ1w-!_?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?!??u???z6?>W??A?~j?t?X?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8??d?`???:pΈ??A??H?}]?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsM?O???Έ?????A-C??6Z?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??ͪ?Ֆ?Q?|a2??A-C??6Z?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-C??6??M??St$??A?~j?t?X?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails????ҿ?^K?=???AY?8??m??"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails???Q???y?&1???A????Mb`?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-C??6????0?*??A????Mb`?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails	?c?Z???-?????A?~j?t?X?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??臨????1段?Aŏ1w-!o?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailspΈ?????Qk?w????Aŏ1w-!o?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??Pk?w??	??g????A"??u????*	43333SV@2F
Iterator::Model?o_???!"???u?B@)??@??ǘ?1?r[??;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?!??u???!Cy?5??@)???Mb??1?5??P:@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??d?`T??!?u	?w4@)???S㥋?1????3<.@:Preprocessing2U
Iterator::Model::ParallelMapV2HP?sׂ?!????͚$@)HP?sׂ?1????͚$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipW[??재?!?X??LO@)?ZӼ?}?1b?<???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorU???N@s?!?P^Cy@)U???N@s?1?P^Cy@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice/n??r?!&?B?v?@)/n??r?1&?B?v?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?0?*??!??|7@)??_vOf?1u??G)0@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$		Rj????[J??o?????_vOf?!Qk?w????	!       "	!       *	!       2	0??(Q3??X"??i??!?n?????:	!       B	!       J	!       R	!       Z	!       JCPU_ONLYb Y      Y@q??W?1@"?
both?Your program is POTENTIALLY input-bound because 53.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?17.0106% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 