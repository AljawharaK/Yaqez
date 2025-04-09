# Yaqez
AI League SCAI


مشروع يقظ: الابتكار في سلامة الملاعب عبر الذكاء الاصطناعي

يُعد "مشروع يقظ" مبادرة مبتكرة تهدف إلى تعزيز تجربة المشجعين وضمان السلامة داخل الملاعب من خلال تقنيات الذكاء الاصطناعي، مما يساهم في خلق بيئة أكثر أمانًا لكل من الجماهير والعاملين.

يعتمد المشروع على الكشف الفوري عن الحوادث الأمنية والطبية، مما يُسرّع من وتيرة الاستجابة ويُقلل من المخاطر، وهو ما يُحسن الأداء التشغيلي لفرق الأمن والإسعاف. كما يعزز النظام من تجربة المشجعين عبر ضمان بيئة خالية من الفوضى، تُمكّنهم من الاستمتاع بالمباريات براحة واطمئنان.

👁️ التقنيات المستخدمة في النظام:
يعتمد نظام "يقظ" على مجموعة من التقنيات المتقدمة لتحقيق أهدافه، من أبرزها:

نموذج YOLOv8: لتحليل الفيديوهات واكتشاف الحالات الأمنية والطبية بدقة عالية.

واجهة برمجة تطبيقات FastAPI: لإنشاء واجهة برمجية سريعة وفعالة.

مكتبة OpenCV + RTSP Streaming: لمعالجة الصور والبث المباشر من الكاميرات الأمنية.

التخزين السحابي: لتحليل البيانات ومتابعة الحوادث في أي وقت ومن أي مكان.

👁️ مميزات النظام:
مراقبة ذكية ومؤتمتة تتيح الكشف الفوري عن الحوادث.

تنبيهات فورية تُرسل تلقائيًا إلى فرق الأمن أو الإسعاف.

تمييز دقيق بين الحوادث الطبية والأمنية لتحديد نوع الاستجابة.

تكامل سلس مع الكاميرات الحالية دون الحاجة إلى تغييرات مكلفة في البنية التحتية.

آلية الحصول على البيانات واستخدامها

👁️ طرق جمع البيانات:
بيانات Roboflow: استخدمنا مجموعة بيانات جاهزة تحتوي على صور مصنفة لحالات العنف في التجمعات الجماهيرية.

مقاطع YouTube: استخرجنا صورًا من فيديوهات مباريات وفعاليات جماهيرية، وتم تصنيفها يدويًا لضمان الدقة والجودة.

👁️ طريقة الاستخدام:
دمج البيانات من كلا المصدرين لإنشاء مجموعة متوازنة وشاملة.

تدريب النموذج على هذه البيانات لاكتشاف سلوكيات الجماهير مثل العنف أو الإغماء.

اختبار النموذج على سيناريوهات واقعية لضمان الكفاءة.

👁️ تحليل الأداء والنتائج 🏆
أظهرت بيانات Roboflow أداءً جيدًا في التصنيف الأولي.

البيانات المستخرجة من YouTube ساعدت على تحسين تنوع النموذج وجعله أكثر دقة في التعامل مع حالات واقعية.

تم تقييم النموذج باستخدام مقاييس: 
(متوسط الدقة) mAP

(الدقة) Precision

(الاسترجاع) Recall
ما ساعد في تحديد نقاط التحسين وتوجيه خطوات التطوير المستقبلية.

