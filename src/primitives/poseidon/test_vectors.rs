use halo2::{arithmetic::FieldExt, pasta::pallas};

use super::{P256Pow5T3, Spec};

// $ sage generate_parameters_grain.sage 1 0 255 3 8 120 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
// Number of round constants: 384
// Round constants for GF(p):
const ROUND_CONSTANTS: [&str; 384] = [
    "0x3b25afc6b806ba2280f989a901cd566620b6a79c90480ff20d1bf70bb3be361b",
    "0x1ca572e409e61023ce9a23c7013f9f5ca5e6cabf4f68f972a3a4c4aef2d0cdcb",
    "0x0bb0bc4d9899da691ccfb7b4df114c1acdf0938da26d8471964e7516f89152eb",
    "0x315ea9b75b8246b97c978fae471e6778e209ce714a04eb674711b0a7d62d75fa",
    "0x39145ff6a42db1e92ae3f0f6c8c523aae34f94eb6a75b033845a74526e931826",
    "0x2c55f6928557c4456bf8ad94eb6286f83c124d0c1f67861850d97bcd0c042dbb",
    "0x125550a7698ce16d01f98ab482d3e3e79145724854e2915f4d235549e7c113c1",
    "0x08a79b12b620b8ee2e0035027982ee16b938f64835d40d2ff4ad81fa6460d8ab",
    "0x38a84243fdaf37d3abc4916d3c96d573331e8a38027fefd2f34a703aa45a3be4",
    "0x1d081967a98251446c60268e5355990696b22a9be9105d5cde6000108bcd1793",
    "0x3d9664f20cd1f3f77f3cffa3af7b013fd3948ae4a79048e716433877d090c3b6",
    "0x32300e25c40f1ed418ef6d4bc235f6e82a290558fe2788e6cef3ced7bd79a8f8",
    "0x072e0e0fc23124ef15691d3d115fd13aa26e01a10d9c1441f826b01f8622b314",
    "0x2050a9e3fc19ac164b1f58ef954eeba1c1f5dd4ccc9e2edd2cdd43b4a0b55a41",
    "0x388d088ef276d51cf8d495b81268aea8dee1eb744537f8d0f0d5dea79bb73d6e",
    "0x38f2d9a8f6a7c176d4ce11ba9b583f91549d78f051801063743a46d9f959538f",
    "0x182c8521cc8939802b0c8615ba8f2b7369b32d3ec3d7bb0441f1243b3ed87901",
    "0x2b1aa900e0a6c9b2fd903a544e81da53b4e3f1e2a89327850add13ad31b37b0f",
    "0x23c01251443952fee267c1bd3b2f8f17625b38c52817593562283a4b4d59f831",
    "0x1fa60e490684494fb414a8a4abed9fddd47462bfa7fcde1a811d6f487b50432a",
    "0x0c47e08bf229b45ecb5803221fac2027f4c11dd5fc2efcb8f822fd91f11172d8",
    "0x1888ed44920e46215426a8847079fcb3c90b323948d14a8c1c5c6d6d76a0940c",
    "0x374d39d85b2b0a6d9a4a994502eb8a44dde7defee829eaf005a36eb0c8f12263",
    "0x182bf90cecdaee91bc94411dd77d9415dd812c0ace98f43ba8aad85334a693e9",
    "0x1d783fbe1b51a2ffb3f867c5799bc86325f11e8e291d76fc726e7a136d1b3c83",
    "0x046d03e8e252bf29c7dc6d1b7aa555c3054c397a37d7f1872b506a158adb6596",
    "0x36d1043a24950bae810d969b9453abf1cec277a4d722a88982eaace2fafb6a09",
    "0x012c90f2e52ae9a7b87c77791666304bd16575468d257c13ab3c01b90c3194cc",
    "0x2cd456d6b0686062657c45850a979c01fe8b8a69b49baedca2f8b911c7871c29",
    "0x2b02192abc660afe6cdeec1b1777b2c58e0190428e7d0014b16ed2725accb23e",
    "0x277449f686cb4312c20dcabb29e8e0b11c3936e458bcedbfc6e88d0455237451",
    "0x394b9a2a40f447da818ebfcaacec1d918764c6b4daba3899e283d65ff72a95ee",
    "0x1d0545a599c9a9defbcf37229c960f27e0a8f5342f2e4e81ed17d1eed4434395",
    "0x2a77e8435bc101cd52ac4001e9a1a3ed422587766c0919180ac210d742c112ff",
    "0x142cb33761f98b61c9c9810b4c4e1c555bc3696f041ac2f20b582b1411302c58",
    "0x1c056ee88a54c38f88535dd500aa5aed4cef620f15a26c272ea6638f14a9fd85",
    "0x1ddd2f847b1cd50754c82bf0da5f7a949971873709ef8db359a009c7fdc4f1fc",
    "0x0dfc01ed2895367cb1d7c92f0980c7615cd765e96dbc019b56a4374a8f1e3771",
    "0x3721039fb53130f4dc87207849ff3ba411acd30fb1deba4adc6b85509d7f7b80",
    "0x35bf62e074fe04a08b7c53114f76e9c417dcfda09363890b72f985b345fb0b77",
    "0x1ca7f5eb2f353606154c68a38e413c90332774da79749de780b74125e2143d9b",
    "0x3c7aea17296b5e10449da1c01499205068a19742e09706c21f2c718c874ccb89",
    "0x0d0f57ad52fb1fbf000ff2eaa0670fe103912bd456c6c9f7f94f9492cca4343e",
    "0x1c832c237a262b5db72e5c7ba3d9252725ecad18e3029a407980510e6115abe1",
    "0x3a91973fbe49eefd36a53a64efcff4e2b52980e9226cfd69afad2e03e3e1a792",
    "0x0bdbb37da9c758245f3646dce034caf209593a38baf62a9ddabe1d2421af5e9c",
    "0x013e1a7709127229801cfb2dec76fa2c57ee54e7d2ae18b3dfc26ede6ffacd3d",
    "0x221c593cf2929d7597fd7d5ab4db42b0b0e9768807eeb5291ec54ca5c4e930bd",
    "0x09b5ede7a2708dc4c043fbfeb596bea31dfdfd2175e0bb35a601bb50d2041620",
    "0x3f88b3af06d1b7574c27993c3707ad9c752243e40e3131f676257d044f2ddeb7",
    "0x26e2adb07473a1541bd048d43b3d9006c35b7e1206baa10ea461950a67b5f1b7",
    "0x0329b0484f4d088411949a6562fa92487adc8c2d74509895c89963dc722605d6",
    "0x3f1228946d7d1cb58bbb39c8138a4ddb9000079a7dbff55d6d000a50fba489ae",
    "0x35c9850f960fb0b6ef9af400bd860e636df9712ae3bc15cd3d99f4c5c2bebb89",
    "0x1616a41da12b2a451f39bd75809a757214cbc63667bbd824770a33b5ff53a6d4",
    "0x203228abb845283e5267be7af5423d94b81cfda781d9727c7feec5083ee80b76",
    "0x252983102ae5b2fc182f18ae6f18518fdac8c9c5bc0136c616f041edc5b45a77",
    "0x362e5014776dcb520742f771b9a8cf59ede38355417b45d63ae04319d5445190",
    "0x1c1bccc56382ec47b00545e3d6ebe6e0afe8882017f5e81e266b6f67bee9c488",
    "0x05ee4754ae960218dc95f82a041137a37392d4aee361d9a3dbc303e392b2684d",
    "0x216a1c4a864b8ba5c96d9b664d0f520490745818627445331f2c4a5dfbfd7311",
    "0x09957bc6b09c43e7d2b4da1071d1a578e66f20617b5286dfeb73870bcaef9f6d",
    "0x00658b3969feed8fe6148057fb99e45ec38a72acfe64151d39c9de50c9b70732",
    "0x3ecaceefdabe0a041b64269d1dd175017b4cbbe69f7793319ddc21d1c764eb93",
    "0x006e91eb4a16dceec706a9eb3d579d7df474e608d8762b30bb90e08cae38ddca",
    "0x39a935c32ba0b466741e6f916755a7b3eebfa0d56c03c65dad0f7f9c4fbb4556",
    "0x2e6845bc8472b9d88693640d6cde732611270f02b6f93a8a9f606085e98e20db",
    "0x066cae5dc69109d98e593759b301f815a6b3d4391f467707086a3ee3d669b037",
    "0x265c54d1002a4434a646804934fc6d80aa63ade30f1933f815be00dfbd88e386",
    "0x129babd3ae61ed94003a52492d5dcbe88fb0efd5c61140aa459c2129bc9c5549",
    "0x13d8d532c9a2141d82f7f7422db40775a95132a1af35068e779b7b7eb092919d",
    "0x0eb085c4b2d5cbabd0db1b88780e1c18fe1391fb8c6aeb29d9cb19af45c02c97",
    "0x31addf291aaab7ab213036b07186b365bb003e02604bf04ac92b80be6313dad8",
    "0x2f313605aba970b6b9c9c2b6675a5e8cf2c8271fce56e983bcf0cb89ad0c2727",
    "0x35f9cba200b39e8fb086f382be1e6ce12382d366b7f5f9aef723c6e487207415",
    "0x3e32941ca1fc80c2e44bff763f2a3b203d367ab00d1b2383dad2472c4bd423bc",
    "0x275381ba0848ecee5deecdf25510b633c3d5231e938dff72f7677d56f738f859",
    "0x09d58904c028783a286dd85b67c02583aca5c393c45ffc22b5a3e00603d7f866",
    "0x33623ff6b22b89e862c886b99fa8139caad603cbe0a8add16dde97b208b68de4",
    "0x02c01de6244a0de6a1aae302dcb745139564f2838e642722de1545b349a2549a",
    "0x138dac2d43dda6f4643682277083a932b7e61a1e06e6c3b4385169b80db8f8f5",
    "0x29529b66a2536710b72712250672ebc4bdee95f1b27ebf0d0da5bda508e438f5",
    "0x10a6b7fcfc5c70accc348edc1ed79f8591fc04233344335c3091f667d96de599",
    "0x0c0ba4f4a28792689ffc278eb3b414122db743d2ea7af72d3d5c10fbc68f5073",
    "0x2d948844366b8df32b85d10e69f66701f8ce646ead799b6b7520c0b062495b42",
    "0x19a723a3cb774d07ad9502fe734c92554a306bc01af9e79152fda8d7e13b70ae",
    "0x2607ef782cac62ab643a1ef315d24d656764d3623eebf168a03e68ceb95d9c06",
    "0x2c549c4bf9cfa472a25987a913715b1baf44682689a85521cd38f44c2399c1f2",
    "0x23420e7114aac926ea41287700be199d6b10c22f6d10b5c9f7c73f3cd03d57c8",
    "0x30d11ebf921501399bd75a113f1729cbf8602e67e73081647fc955dafae7926b",
    "0x349c13d261e729b2bb4187decaf44d2cdcb18664d2fe1b43a8d85a824f46b78e",
    "0x12aad04eea5c528055c42c79bda22f1132354f4e1b90b6c67d194caa9969ac95",
    "0x0d2c4b793002c11a87e0074fce3bff211680a86d76ca650a87885b32d6eae5bf",
    "0x3e6aa80af3d73d853ff6a0917df15171f1b43be1af7a4d10f24baf034dfb7779",
    "0x0e3d184eacc1f17c74256f928eadfe182deac5be4905a27858f0d4862f510de8",
    "0x29b1e4fcebad1a799dafcb7f06f63ad33d1a1df03c24e868e761ec84c7927fca",
    "0x081c9cbfe10a22357aaf892e58b84ee63f027613d70fb1fa2ac0cc354b08483c",
    "0x1db61d46c32333f251d578fc59e050a4162137f82a8881de38d2bba5992f6d99",
    "0x3211b70caa0df6c97503f80327c590bcc0f6ad6f93704ef33b5273d6144ae047",
    "0x025ec2e141fb4f4d478e278b7051a9e9a5f3daa4c1fd1d6baacb468de39357bf",
    "0x0b63e478c0c724b0ec1eac570a09a1b926172f0759a9cd8c865dc387b66696f5",
    "0x106e1aa25b4550ac8af35685173e6f394a9904347d9b8b298ae8f838f6deb884",
    "0x24d3e4e6c7cb87e8f0789e268d34b3494d9a1a45759dc6ad5e85148584775a76",
    "0x00dcba9bf715d6a14aa385927c1d3264bb8e751010f239d6396226a5e1e2b327",
    "0x3b637889f503cffe149e2b6bf50c18f4cd0c3cc2fb7260804d3a67571e5fc5be",
    "0x0c5ab5d5b4f428e6cc9b263d88afc47b1ddd449b5ee51ad6923764b3ae92a9b3",
    "0x0e456640ed73c584859151cd98564796a8bb28509db23f315ca7b7ad1daf22c9",
    "0x1e43865565fca57a80a2c53e252d9cb2a1d64a49dd54128f8bc641d278351408",
    "0x1431d0549d041f34cd6cdfb426f885632865563c3bd19ee4f2a7e4932b0c7e01",
    "0x1c76185e9713e28dbc08148b684f8097e2c5895ddd1588bbad3f1fed4deb13b4",
    "0x0dc6f197e25304795e81b4a56a64d75e25ba2726cf5f35d560bc0560ac3ac7b2",
    "0x38b636a0f5689764ac958b43385980a86f54b4063183e73952b7c4f93a804f70",
    "0x01bb61bf50e4236cc30c7970722adb2363e1cf29e268cb7417ad80f8f9c6ac85",
    "0x1d40a6ea86cc01c9c8ac893dd03e7660cc4accc5c36e629fab77fe5617740f13",
    "0x1219a127a16f4432cf2acf327fee2e08aec997636041a9ec6cdb32cc7634f11f",
    "0x0c545f1e3e4562b69cf337181e38e6e34b41785226a85bb9590116d92e106683",
    "0x334d30e573a75f679c47f412a1806ae2332cccd9d4d60bdc20c58d460abf4247",
    "0x277b15322c63c87a6e81f1f596818c35f0fbe0ec17aab0e9a1ea94946c5fe98e",
    "0x1cbafec6cc9b4084a7abeeb370c3a6a829fe8042390d56e3c67017821faac1a7",
    "0x3278c24b3ed150e6a2f534700068354803c556e3a5bc9e0e8d11ce4e0705bb4a",
    "0x3c202e2ccc98a96027158a5410a9e59ac7e0f982df14b4a4390d8b94045fcd97",
    "0x361beb0fbeb0e0a99173052730922b1faf836b30d592404dca8098c0ea929506",
    "0x3423f54410937cb1acddca24e39c8df749a407c0401cd74177f77cc6668f4609",
    "0x31abd046f17b8f129abd12b0eff20318ea8617c95b43eb2ba2d060c96496f818",
    "0x112db4e149c93dd4dbc3b9d8c4c1c159d36b83d9ccfdea9b004a662e4e9d58cd",
    "0x0dc3ae49fafecdb4f61339bac54f6ead9144814cbce0be03447e640d33b373d9",
    "0x0b6cc8aa6f24d7d86ccfc428c8495a95e064b38d7f6fedf80837df9b8fcb1fb3",
    "0x097e4e0edc7267dfadf2b4d718e175f90566a46ad0d2df560267d0d992243b13",
    "0x3229895750ef78eb656ce30c31e0509aefdad5ce27b65dac9218860fb249e106",
    "0x2adb86116cf72585dec5ffcea770541a281da3c0667cb9d92da5b3248e03639d",
    "0x237dd4af8db23319749785ba067c59ddbec3dc7964b1f086ae9c9b2b184b2880",
    "0x283140df5881e1a2047aba68dde32e5b8d35f746506c7aefcc624abc65282ec6",
    "0x02510b294766697951e2274963d2d9908e6df2209717a03f6740dfd09c4bf001",
    "0x10e275d7a304f126af3abc05eb39e2e8cc8f5b87f5dc4ddfd1daccb1f87cda15",
    "0x26c23a1ec496e727b647f2eb7e950273e59e1210df38ff9f5bc7e9c31db01eaf",
    "0x324175772460dd1be78908031052e86393cff81d1cb5e60b6c28d3706688626a",
    "0x13bff1786447d1fc7ef79fd5f0be5d8a657f6d2cc96fac1e208bbbe8720d5869",
    "0x030db325766741b2cfbfbe508216ca508b3bc9e85ffc901f0e167c5c194f2a93",
    "0x051e52bc595e584516d62ed009f78430d9eb753d952ca694525ab5d1a936456d",
    "0x143ee99e903dc3c33f3a9b6d547b5ce1dd74523f5d50f005b20094faa17ea55b",
    "0x18b5d4253c4f2455001bc9d739278571d694ce002140b980570077532c029a26",
    "0x3373fcd28ede7ee9c40148df6427e611aec6a67af878ff1fb5ae6d8f391407d0",
    "0x15a9d4523f1321640d99cd913e8c4586dfcc1e811487697f40ef8e10b6933313",
    "0x06be89b85f351c627880ed56afd2ce646094c98697ca7e9bcb8affefe599f105",
    "0x347cc7db7bd520b4a663123c80e68bb5e283578c43d3b6fd80dccc6e5eaa3ea6",
    "0x357c60818fdf132d58553e62c5aaaa2755aa2bb505fc8d387912c249fc1b9e7c",
    "0x2eebddf4f92104bef78aa15f9f059bb4f433306ad764eb8e91f0cd5bf0c74b5f",
    "0x172b9a2df6729c3578983be9e0b4e04f3752a0d58f525b91f48b87436c0e5c09",
    "0x00447e6d57f8caaed4052d6b8377ecfe5232e8ad4eaefee0c719a57b2d1ff8e1",
    "0x08b65a9ae68013aeb905138852555ba3f04f531aa6176a6de4c3a7eb8803143a",
    "0x21d9819b68735d6074a8fff3a4df6963b39f45f5bbe38121b2d38896d48fbd49",
    "0x21bbae5c35368e29140b84752441255f6dfce2a8e7863b894d32f34f5423d50f",
    "0x298a88e20cff2d99412f633b5fceeea6628ef7567c9af5d3e5223bdf1fc55bfc",
    "0x11910be0b00d0defaaceed74a422331f8c12af60376e6e0b34cccdbb5f705d63",
    "0x0d68e4ecea079067e9e5861da93ecf8ca8426b91a8a19046df714161af55970a",
    "0x3865e1b9df6632bbfee70acf6c097e489b06b473fc6871b331a8bed10e147383",
    "0x3a126c3f24ca72cabaea23ddfc19561527824981858675dba75998e7188a9dfe",
    "0x180cea1c27d530d427bac1d5f731f4be22eccad5c31d47ff4a86f202d43c72dc",
    "0x13c12029f2a0ba6771e1d0ef1159cdd550b05380b64cba2c58cd40b65229da36",
    "0x3559eb0a08be1761e25c67824b84609869fafe7a1087677d0918f3af55ac07a5",
    "0x3984ee17d6d266d11863ba10da69cbb953836d3245da350330f87d05d06b6a6f",
    "0x0c75d7dc2743afaa2dda175003935df3615841291f4e9d08e3b91812d88acb40",
    "0x221857498facb51b05ec3f122aea0bc57380034d1fe80d47267b9cb9784f5453",
    "0x37b8a30fa05b0e23ec7947ae3ce1b442a141d60b997925a9219858450212c384",
    "0x29f2b30afd2a557423d2020e0c77cb443d3433ee6addad0010efb1ffc430961f",
    "0x21cc275f6a5603e2389ef30cc12c702c0b3cd20f14bb17cb07703bc4d3ccff27",
    "0x389998bbccc0709b715d96f0d329e2bd0a80ae5c93f6eb06a479f748ff6abd0d",
    "0x18789b5b0a5b6a4c066857852a794280813f57e46d07ac74bfd06b1c995fb345",
    "0x0e789e00445345bb701d100fe46903ac95726e93edda85ccbd6f0a1fe430c196",
    "0x29f117a18aa1b9dfbb1cdca1c1794ce065ff42a6794c64165a93e2c369c88da3",
    "0x3b9bddf22ead996f154180e2e8504ff68762545ace38210961d60c01dbe3dedf",
    "0x03d524b9da8e7991034d61981893ed35109acbc8882017aa4b7077b5ed37345a",
    "0x39f97997a306ee4c6021f5e1e9d8c3684e63affd9b7c13eb56be503241c21f56",
    "0x03b53516deeacd1d417ae543d9231aaf4327266cb98d967b07bc1dba7e9685ce",
    "0x037546f27dc709af8b1f19d5cd28a6e50329c846f2c9f693b00aac73e6f1dec9",
    "0x0a35ae8c59919f47f74b6cf56c73e226efb2603f5626decbee5908f6656158c4",
    "0x00f4031034366c12ca921a7fa2648eaac4b856f799f6852b95619df736526b00",
    "0x1cdb37d167f2e1921892623d4e54a578dd19fe1148fcb73556beffec96701f4b",
    "0x04e4e5f8e03131b9fc17f08eda87983b7a7588492dcefda4d341c0258facc8fe",
    "0x310e77f92301471700762cf83141c102552598f26788974b8667e0d2e9810c7b",
    "0x196dcce748de2dea69bac33fccb84c104b62556663997ee9b1e0e1beb7ed040d",
    "0x1f7b48acc67c02399b6dc91caf627faac82cb608219027f66f898ae336336cf9",
    "0x240286128f8f4c1dc8cce4146eb53ac4b92b994763a2f07d830c176f60c25b4b",
    "0x260b8c212416db27221ed79caa02e9641d9f255a147da3f9f024e84839d744f0",
    "0x28954ad05bf7e527d008f9ab6c496bac21c5f3284640d8305a402918e877abf6",
    "0x00afc2b190f2052dba777113c6c1bd684e50b05498932ed378b52b63950a481f",
    "0x3c6cfdb4b65ea0dc444e990a1eee4c074f735c952b1fba5fd42dc303f46aca82",
    "0x1a743696e2cf82a6fa14646ed1db0d83a0ed6b7726fdb2e11c5c8df46fb18402",
    "0x17ff036607e5976f8ae6f8a207c20e2472f57bc78a54dc350ca0ee23fd53a7c8",
    "0x1dd362894674bc3041fe69c447b7f42a425e1be4295840a4211fedf1fd0a468d",
    "0x1ea2dbf057eff71176712230da42618192046a5f4dc738f85eb7d7544dcf4712",
    "0x1c2c18b40f88153529bc83e9dab3ce5f35d8658caaf80acaf68b2ff7b4d8a3e8",
    "0x24aceae7f37241cb181dea9fd114de69bd9fe61243248489530a6175b170cffa",
    "0x2a17d8b7dc4f187d21db297460b78fea835ce6255c1a1c4dccf8340237d79740",
    "0x032a7eaa1c41727774dd2453193971f7fb33fcaa4494886083ab4ca90bc4ff7a",
    "0x3a630a0f126051d64c04214051b812aaa17be7861f1a49996110e27b9bfc7876",
    "0x223135578081c023d1c57b1c2a7f90ff4f5e0530b7840c36e1cd3bb237c69d35",
    "0x30df4b265d1508944d48f7ca8b06eccdb447b4b5ab7d9ef576a11ce4553d49c9",
    "0x20778bcfb69533241f8d18c9ffd97d699c3183ca4714998568fb8eecb43d2c82",
    "0x0569481755623539c6f24fefaa42a44c740c4c948f7d70bd8064fe0dc67fead8",
    "0x3b7eab4fa6fdf989f8359e6e711fa820a7ae1577c2ea798cfc1c62ede390dda6",
    "0x285fde50988497104c3bc2bf54979bdc29b29cd845ccf73b5ad08a1cc986772e",
    "0x07f87db44809861f26d3e9796190b5cddf530777ba8c1bbfbf98ffd907103c3e",
    "0x04a9e29a7ce3ae379a0930b19316c24ba2201d4860d93b9d51c709c5a5baf1bc",
    "0x279eae1344e0804c51c22e2a53bccecb224ce4c5637c643bf9c079f6532856b2",
    "0x111fc0f558ce0babae9f6fb4895c771569669499ef88900f914e035fedd0ae6f",
    "0x1f864b7e2ba17ef37b862ab76c5d898e0b4a0ab3f09c00920b1e05cad5c61e7b",
    "0x32102b2fcae7609dd76b804527661f2576bafc2c8ee19d94be014c8a44199862",
    "0x224c0a38a87accc72b968482c5c18cab0223a83b2ba78a01965fe0745696951a",
    "0x2a7c0807c3f7905bb3596accbaa8a8cceb107269b400d5b5cc069dd33c236297",
    "0x370fe259c55ac676f8c5bd11ec5a21f82f7a39c538a97b09ca9ce2195c315bf3",
    "0x2339b60f893856c4a74eebc02e04d7f4061071a219e7312bd0d86151dbfbf64d",
    "0x0c03e48ae8d6ef5645dd73292e2949279e706f8caa302ef6844c05ed9a19d82c",
    "0x057fb341cb724a5368e3fb3afe9c4b2fccd6f65641a67f421b1fc20a4d25c3a3",
    "0x04e4b91b9665f372a5e99ebbb86933f8d0db469ee5fddf85df69c5a2650c8ca9",
    "0x292a64f04268a1c1de2dd8e3d84fcb59b769b63b1f9de9050ff254ebd16879aa",
    "0x265040d15eb0f67ed465720edd710bfce34a2ab8f733062e67335671f9224fdb",
    "0x18b77ec0ed903b6b2a95c9071376711c988a7fb7d18f2e9d17047b6ea0fad4b3",
    "0x3c8e16223faec5bd79ac1643bd0b9f8924c4465239c66acf3702fc362dca4f1a",
    "0x2817095b9d166c7cdfcf9f6875d4467af4009f795c98e50ba8ea7d0ab16d6137",
    "0x0dc04c5ed3f29e2ca9206f2dc17f693b40a5d76ba3f6a905c109560dba770808",
    "0x1e131740199e94901125f0b45e53e24b4553ed75be35cbedccd0dd48d1796dd5",
    "0x37eb7eda1f97ff27f665f1d77f456b5c88f86e4638bdedb5d525223d78d2ccaf",
    "0x0a44ba057811dfb9acc860cff41472f3c4d90db3207ec89c13d3969d9a5f8d22",
    "0x0e297d93f8a1ce58607cc217d84a8bb941e9ed7fd1e501a99eb9228046e94558",
    "0x03b9fc48a79bfa0aeb893a2e01b9b8e4da994f0bdf574b313ba7fd4c1bae0997",
    "0x11faf187b0a71a052787b0e907e863ffcbabd61cfc0b62f2617545d69c577c75",
    "0x3e7662cf72761f433a071aa1a726b2dabf44ef3c322a7c16d97d55e1619bf6df",
    "0x2e0967da554ac71832b965af68ca9d3fd9c61dbf246cef9bab350312c63df50c",
    "0x344362784b933e3f611a3b9f74b11d0298f549a48eb0ee1c0ccebc688fd89be5",
    "0x01ef0ab762967970266fd1db109d140ce76cf4945be917892826e510b8850942",
    "0x2dfbaaab4390693cff25fa8ec090dcfd9f5033f0b8183ef1f9c700ad298bdd50",
    "0x2dc8daa1fe50070c76aec30eca8a156a355969c2fd062f50381ea9d1d42c5706",
    "0x345329f22efe2b89d3823dcb7244f57441a4b12a7085f035b2bbc42aaf5545ce",
    "0x29880de424df77837a157629b56e662d252a303b17f4f0f39e54fe150b467e89",
    "0x16f41331d8349c1819752149b814edd10adfb0ef75a2e7d911cbb6ceaee464da",
    "0x1b632f8a0e91c88e922114afcdd365d3941ec1dfd945a9f94ba5d48bd751f01f",
    "0x043dfaa7133ea16e68623f1f05623d3348ba86eb9512614ea24cbb03884cb5ce",
    "0x276590c19b98b5104ca2f3008c26aa7c91bdd643507156eebaffdcdfab7504ab",
    "0x21d76621271e552fc5e0f9ed7ad6facd264df3a069666a6a3b0b2df50c4b6c39",
    "0x1279676349c5220ff8d3179636342bce14e9a96963718b365b47b80e6f5e43b5",
    "0x24b7f87dc5d2318a1fecbc6fbddaa7b017b7b9f609e9ac6824583f312e75a725",
    "0x31d43da8d623198cd6de5913e84b1e0139e577562968793eed9fce1e11b04e14",
    "0x19e7973efc8e3af779384ae59a4c86f160328e1b3e7350bbb4dcfbf0b4c21997",
    "0x2f0edbf63554e02cf42607947cb5779d5c4438c52a937a726b7a1329c810c928",
    "0x01985ac7a0cab940b9c07b9acbe9a4bb4711f1811d2a7b43bd7d5e8615a0e6b7",
    "0x0165f1deaf5b0701d5e510d932bea7ffa139023af0cac9c9722e65917850d613",
    "0x131aa0230738dc8837c87f64115be67b947e094e106754df54c2f789219d7168",
    "0x1e1611b0ad59eac320d1770729fb640552a2113c3f3c651bf378b24b8b276fb7",
    "0x01280823fe626587623e26149378349e2a09c3f06507bdbb20c12af11191f5f7",
    "0x3f2b1292d61282dff1655be6f3277fc33df704e08cf057d1633debe5f55823f6",
    "0x35acc28d66cf62ea50444fe43fc5a00e8604dda7c89c897e3e393fac4419ed6d",
    "0x241a5c3319a988843b7817aee62cda7a2887afa2bf6472869dc08ca61b904135",
    "0x2b7c47230a703a2a0adfb8cd47d9b26be96e0c41885ddcc5dd1bc9ccd4a1da6b",
    "0x2ba2680bfccb3c0a2f60daf98cd07bcc40498a97c91bcfd4d105cd61811ffb51",
    "0x266fa9fc40ce96e0130845f64dde09df87bd8011e9394d9039e0331060d68e7a",
    "0x1c1ee5784499c7b3e4f8abd0f5a3a5c7e9e99d14c47144943ecaf0b73ee27937",
    "0x208c774079ef90fc5b1f35f4a1d4c98b87975c1233bc91c248d0024dd9f082f0",
    "0x283648794351f363726f208741f0b5542e2109845ecfd150202f6bef6e5794fa",
    "0x2b5dbeb182b937307433274e3a3123391a50d3c9453bc5b35e22a0da11775fa0",
    "0x2b2e5f63da3dd000971598b57fdc04407f1d8e368d9e15128ccd9317217d7103",
    "0x11f06300d9da0dd339d964982201289692414781c1d916602dd08bedb2194a4b",
    "0x327632aab14c1d8c26550b3910ae610c3e10fc7f7f14f430be65b79dcb9fda67",
    "0x35e618439294a33a5e556715a67a57d3016bc1901e236d2f94822a296ad02c21",
    "0x2b0ee60eb42ddd4fbee35f93b5662fb5dd28c997b1a542591987e79c28a364dd",
    "0x37eae779a5c46ef78076bccae32dd99c6c08477f9f05fc8740141ca3a1efe55e",
    "0x07aa21ec2e5a93698c69c149fc9dc949b8d6de1f8929b0887c85b646fcd749b7",
    "0x31067399100e65fa6dba5856bc27f13c7eeb66072ab965eccd0a25ae35b9a465",
    "0x19535e9e6b084c98d79045b71afd75241666a832f27f4f9b79bb52d7a4c641a3",
    "0x0162457a674196b8b81efded1e21e750d839f3d90041220e41a676a1b04dbcd7",
    "0x14c12827992012948d21c3cac8477015eeddd121891907b1666bae7b1b97b045",
    "0x0b5cc9f00bf9931b9bdc50121525525d06672f18f7ac8d38dc6a1f34b42aafc2",
    "0x12531e7a0b2b25721728d84770c8f70f2585ab5991796293c9c8f5534268a9ff",
    "0x14af0ce03600be8dfab3ee5052d127b099b187f5e50eeff8750822e86a1ac371",
    "0x020ec4b17c9bac9a4b65367b2088950de53f00455c2688c6564083c36a31eda5",
    "0x226c38feb960ac5d521a36c47712e2e99fe37f4013d1e6dc559416897af8bef0",
    "0x1389dfef61183782fb24e2d6285197bf1972aa208ee93ff32e93e0ea60819759",
    "0x1abec9e827a9894d1dae8ab47d655f966d30efc289b9562a4f93980427d9dae5",
    "0x204b10933a538fa91516cfbd7a22b4e1dc8098e4aec4a18cb778a915681a53b5",
    "0x37a28db1799ce6d3f6e66e31e05866193526769c3cdfde6fe6d9618688f78b30",
    "0x17e3a2f6e33c55b9c037f673109701784a44a8987c70052c9ee1549dceba8454",
    "0x04591f344681db62c6fb01b636110fbc1f8215eecf9674391e974b730be4a79e",
    "0x2f1d9f74920c7dfdfe1ec52e151907a427412c822a5ecd00b95aab97117a7eff",
    "0x098443a134a7edad8fa35c4e671bab02c7e36d9829d932c72ae416f7352c5dd7",
    "0x01b2acef6eadcf6000010671ae9444055b4bda5d57df7ac870799356c393492b",
    "0x024710c497e23504ff45fdf5523ac1bf506745739925c64cfbca1af577949ff6",
    "0x0e5d75788b63efafd813415a5b790eeb35d47e7edfef67017d7467b351286e3d",
    "0x0701b8e25056fd9dd0abc5f5458f4a63155da502965d9486b7d48d9b5f555967",
    "0x13788f10b7eebe2f0620f292121b0e9a3954df55fea4b816a7988f6da3e52b52",
    "0x2a9bb80787ee07d9d108bf7e89599422fac3ae6af23a706dee2ed83bdae4b82d",
    "0x087b9727351c2d4ffadfbfe1b74f02b357f28a2123d5c3a7dc99f46edd198ef5",
    "0x1ddd84211e96b99944a4eebbecda044cec7f4fd952f04961832e41d099299000",
    "0x178dd8ca8a45d9365387db624c93bfd67cdcf773657f511c56e86b26f5e3231e",
    "0x1bcfebe866a2822289993f2e54fdc4f9095a70c1bd8000d18be9c5a246df8a50",
    "0x2143fd68da28ffa754f35ca730325181bf53cb8195e8385176a199efdf310595",
    "0x02544c79a41361abeb85bf9534257613581e715768dc6e9a709e2bd204177513",
    "0x242e9800c7e0c7604f13d52f6f2711fb6b3f80c60dee324919cbb291323079c3",
    "0x2a8700e62d9fffaff566e1e2d1dc3ea3a9e55f5f87d053799d528cf8cbeeb7a5",
    "0x32f9977f7e12a981875205e4818e763ef63bc404e89f7ee76bb8d72c8298cc2f",
    "0x370883a160173b4729e86aa2dcdabe47e028ccedd4ae2f539eb9241b4423a724",
    "0x08085d18be2f11f0921c6a06f738100becfeeb491c7973b4e617000881886612",
    "0x09be34aa60a3fb3ed781f5ef5a0fb1e90574a6bd1fe86a97e162b76cdc60f2db",
    "0x239aa1a52f750cee6f597e440ab3da5d2151116fbe3957f9e0d6a6361eb985ec",
    "0x20b436f951ed4f85d20f224ded7bf8004c53908cc6e6ffb142bd09bd034b34e4",
    "0x2e0cc9ab55cf14147f2b2014dce721c27c8324aed9ddf4da3ae6190446081a2d",
    "0x07c80b20f4c97c2c1e8d16787be494e4904afa8040c7d7228ba25793f2b99359",
    "0x184b20255bb1dbc619d867a40d04ea4921a92fdb2981d1f283e0c742edd933eb",
    "0x1b4cfb0b8560b4474bc18d14a1245d6a6eb936f469c408024591aacd0b7ae831",
    "0x0ff879f0e6e7cfa59c0ff70bc2f11e58f4b9229e99a39de3c22d79285ffcf10d",
    "0x0a2e8dec328de6da21dbdf86a64239587a7a1c5f1370f29dba20926776bd1e00",
    "0x036457fe206e1f1757985b97d20020fcd9dfc2fbc5df6d8756cd0608572487f5",
    "0x2592bdc604b326122571a1648c2d8c4c632153354ef1011ec3fb3f951d26b758",
    "0x187f6606967b6ffe882101c91732a515f594672267dcfaefdcb9603480e22d13",
    "0x080d30619fa7e790926e75013f54a4b51dae54ef9b96280b1f4ad8c0a69f7600",
    "0x0087912a381f23e23a2090df67659631b516377653ff97508af98fda2b1b6c3d",
    "0x0a6c113a8665cb4130c497f2726f8759cf55ce4c962deb80c70615f6881fcbb1",
    "0x30a3861fa791b625509d19306a12eaaee42a15963bb361f1a165849c8900161c",
    "0x20cf9d64da66d825dc780da2bb2ab8a358ce2feba015c9e8100556702a6b6f8f",
    "0x32036057397e0ba7ef3362a142fcdc03dae3d35a1db31a807632e455941e4971",
    "0x2da8711eead959f3a2777f12cd2934aa08149fd120ca75f2779cae8d612c69f1",
    "0x1bdfef01b6ba9f5a54f9a8dd1bfe219991921084de1aa0f1ea135daf9bf7130d",
    "0x1ffd4f3ede7b58eb8eaf51905ebf82f52c54136782a233ebe18dcb0f0ce7507b",
    "0x2369fbd8d028ff73b8a3ac9dad1b297f3b8cf6e50229f545fa369bf5700680cc",
    "0x10f04827f033af8207a499193b4e55b5834e6e6d16787e0b85bb4c30e39f732c",
    "0x1ec43bb8e73e465a4f6f8c7ac46085dfb1a3ace80d46ffc897894d38c7926cd1",
    "0x130c2071318792669bafdefd3be3eeb46f3adfa9912b7df4c16875fea3e23dc1",
    "0x0991a7139ef8cd18797544ac05b9becd46528f877fb0e8611c65253e60808377",
    "0x26e2369a3d8e8c56912192b93e1e91374be8920c6245db56edffb0d053ef56de",
    "0x20f7d05774e13b0e6fb96e19b70925824991d4c160869c167c3c9a11f90f9675",
    "0x2e621c4c5a14c5691fe6b666c1a0d1e67e91d34f6438b538120c1cea9c294867",
    "0x15feb408c525b866c02100440ac790bf1230a44f63866769b52dff8f6b2211c0",
    "0x239479caf40afe22649498d35ce0f18ae0c287849624dd978ee6bb53c8f1b273",
    "0x3312163c0c00471d825e4e10be1929a8f40c1af034d979022de1e87ab6d33f05",
    "0x1031974f16f30a33deac5251f06e3df65121a8877996e276782788bfa5e24f92",
    "0x3245758c93052f6ff90bd31beb03b99e2b375e66b15c10dea09ff5da7e52b439",
    "0x13c5f825449039749b17d83159fc41fce33a4fe783840d842aac44eb6ac86a4c",
    "0x21b01f6fd1d6305a08c2a6c217c136412a8388e10a40ddd9164427671034285f",
    "0x17df7a8418f6c1f75f386d3f10c83c8155faa3c6b804fc5c97f7f4fb0931e1f1",
    "0x0a68ca14cff41d36ba575d21ffbc1fa5728c24a748bc30b4df0f6740fe7f72f2",
    "0x1d97bbf9dd69e78c337c4b29bff4ee6610cf681425710b259a01e05e5c9b0a61",
    "0x27d118efa27a7bb62bb19340eedd40c611d153bf34713abc5f7fb68f13d5342a",
    "0x3684a44627d62ed72517b9bca0b207530f6a5f3f040b99a831e0f346b490c1ec",
    "0x160809c0d4f241089ac92c38573dbd16905a79a92457dab89866d1660228893c",
    "0x3bd918b1638cb3d0c4a0d6dc32a38a57130602fc9dcfa5168c6e32763535ed7b",
    "0x2d21b744dd3c32209d9b73e8c432717e2cc79647531d2c43949774239da9aeae",
    "0x3b03dd62da0432e13138bd54beeee16ff66d4d3ac9537f73addc605d4ffcb9ef",
    "0x26e921fbc318fbdc529c08663a7387afde086a2b18c9dd7efe698472f0025362",
    "0x226bdcb7a0c434b7e0c778180a511e8f66318b506338b74046daeb6d84fd9007",
    "0x34f40926fa043c174322ce7a2c6457069f8a1bdc2d99b3eebd3c9c029512fa4a",
    "0x2273c82ed5fab577d4aecd4ab5a9c7761b5641d69e67aa2e7ab5a6712d8934ef",
    "0x1fe8c90cd6876930d0edfa512fd9186f609f56ca4b4e7568f8a7992166ece88d",
    "0x32e8de1da82a245b15ea76a9b7983d5789b2acda372be7d91c88e8f07fe4c024",
    "0x22ecc6f8a7a343de7635c269462a7bc89f5d6fa0ebc9292028cafe976c8aee62",
    "0x077581fb1d9a63ee9515826a92f723f7d14f9f6264f5e62b837def603d1fe078",
    "0x093c0404ab2b3c9d2796b04d0e868643f9714d74b8773a593c5e29ec6fd3e202",
    "0x12e584f660981bfd055bc25c1ecbd9640e3deaebba6717f54826c837d549de7e",
    "0x1458a2080bf28c54222596768bab440ca93f0b9e52a948d6db3221a9ea0c80c5",
    "0x08e9795744b32dc5ae454120280527638c2e9f3ddc2fcea07583f7e64246ff48",
    "0x0de714513590392406ef3f9f89b18f5a62ddd7c42522857a3ef59ebcfdee6ec9",
    "0x3a3dd909b43a8681d0c5c29a381decccb05994f0ab5a2265eefe5d4d982c044d",
    "0x0a36306b7cc9aaa1983d6246685844c1bee6e9ef9ad2b643fbe4fdc5098e138a",
    "0x06bddd9564c2e599ac1d8ea59ae33e0c0764a8d8eff2ff92f8c8204dc703f81e",
    "0x060e0c65e2edb56e49739cbd1a02c830c3d2ffd972127a93efbc50dea525d8b3",
    "0x387f6fafbc741bffcc13b915b084338fd4ec81c7e1a497f16a940eb10631f6b6",
    "0x0cc72f9280e7be71312c244a8ce7f8d0ba142769bd49068392be63300b39d1db",
    "0x3bd2cf836f83950eab8ac918e0581fee0bc1a2c9207dd5b38cd0c7e21ab675a5",
    "0x23409162947509632a809619867846e6c24b889cc7dee00b3b8fc5dd4ffb6a31",
    "0x379e37c59ffe759358f852adde479c7420ec7cc699e95997780a6a62265742c8",
    "0x37c3bdecf6600716a0ecef60b221cbd0e6a32a74cb7fceb49f383593c83fcef4",
    "0x1527f234d25e429e25161e144cb59498bf7d19cc42495251906e69c52aa4240a",
    "0x1385df74a6161c145c604cb35427731ff50f3904d4879f1e2e0cbfb62126b176",
    "0x15da18575787da934bd8f0c6e6ad6dc15267fb1c4020329c232ef9fd0186eed4",
    "0x1543d41d31db3b440c5b140ef80d0cce4953bd8777060b65eb652ac306b72826",
    "0x1ef832584c62a6900dd31743582dc7f134588de0e82fb5d6a719b18b84561e1d",
    "0x1b8b2a0be21f21241253ffbc4ea0b1bb4cbf76a833054563252d58f4b7284049",
    "0x3bdb4d9f14dc2839abb24218836a4eb1fe5899635803ed395d47ca24a9c9757d",
    "0x10beadb527ecbc1cb7348b24d6798725a53cc1c2f3f2de3f8b9a8a546cbd0e08",
    "0x18c14769d788f3b855d39559590b4c4a559f1a9fbc54b8c6dc9dd28d0d25620d",
    "0x1ac30c6459ab5894739c6ccb987a07857d421461418e038b8eda9dd814d763aa",
    "0x2683e5c655df4608df8bd9a1dbf484ec9a30cc08dca6271a16013b0c2888addb",
    "0x0495ef56e50921b5b9387a81e01e69a0d0c4d3d75cf454dd2c0757e08bfb9056",
    "0x029b5c9a5be8401ebe167337c3bda12ebcfa6adca3463ce9008082692a93efb8",
    "0x11dd07d98a7c925167d62042420f9666a0de02484224ea53431a2df9eeb5b991",
    "0x3b3a49f8f83f78f4afcafe9565b5684f3d231fda0aca0f8c40019de5afb66c25",
];
// Secure MDS: 0
// n: 255
// t: 3
// N: 765
// Result Algorithm 1:
//  [True, 0]
// Result Algorithm 2:
//  [True, None]
// Result Algorithm 3:
//  [True, None]
// Prime number: 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
// MDS matrix:
const MDS: [[&str; 3]; 3] = [
    [
        "0x0ecf35c18051c7281db313539052bb05b9ab783bf62eb010ac4741410be2252f",
        "0x1bf012d326a571662d8e3bedb6791c13fa0913b479d771845ba5a181d8c04a82",
        "0x194f0dd87fd22f953d6b5d6b5854daacdb26c5ea92d72fe53991cdbe6f474abb",
    ],
    [
        "0x31e91b394c132c69cb7c7bc430ebf58157a4096abfa28353ec1773a4483e05f2",
        "0x392b9de453006cbe7218f87889ad6b16d06e39218bd63088f49e99b5a083ee8b",
        "0x1e96c4bc4d8b8c5cf93662d5da7c4d9b6c5c3d1bc5eccda12c68c928affb4f27",
    ],
    [
        "0x32c076ccff5138fcc55606edb3c25b9ebac93d00209054eca8134376f7b535d3",
        "0x05943b16bd644cc63f025e3d972339da1d8fe3a0f2f614588e0017956c1def49",
        "0x2cc2c1dcbbaaa6a9b11c5c4bea809bcd3a2a4aa76efa92ce31cd0e59b63fc58d",
    ],
];

#[test]
fn test_vectors() {
    let poseidon = P256Pow5T3::<pallas::Base>::new(0);
    let (round_constants, mds, _) = poseidon.constants();

    for (actual, expected) in round_constants
        .into_iter()
        .map(|round| {
            round
                .as_ref()
                .iter()
                .map(|f| {
                    let mut bytes = f.to_bytes();
                    bytes.reverse();
                    format!("0x{}", hex::encode(&bytes))
                })
                .collect::<Vec<_>>()
        })
        .flatten()
        .zip(ROUND_CONSTANTS.iter())
    {
        assert_eq!(&actual, expected);
    }

    for (actual, expected) in mds
        .into_iter()
        .map(|row| {
            row.as_ref()
                .iter()
                .map(|f| {
                    let mut bytes = f.to_bytes();
                    bytes.reverse();
                    format!("0x{}", hex::encode(&bytes))
                })
                .collect::<Vec<_>>()
        })
        .flatten()
        .zip(MDS.iter().flatten())
    {
        assert_eq!(&actual, expected);
    }
}