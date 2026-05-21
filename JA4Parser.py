class JA4Parser:
    @staticmethod
    def parse_ja4(fp):
        """Standard JA4: t[ver][c_cnt][e_cnt][alpn] + hashes"""
        try:
            parts = fp.split('_')
            a = parts[0]
            return [
                a[0],              # protocol (t/q)
                a[1:3],            # version (13)
                a[3],              # sni flag
                int(a[4:6]),       # cipher_cnt
                int(a[6:8]),       # ext_cnt
                a[8:],             # alpn
                parts[1],          # b_hash
                parts[2]           # c_hash
            ]
        except:
            return ['err', '00','err', 0, 0, 'err', 'err', 'err']

    @staticmethod
    def parse_ja4h(fp):
        """JA4H: [method][ver][cookie][referer][lang][h_cnt] + hashes"""
        try:
            parts = fp.split('_')

            while len(parts) < 4:
                parts.append("000000000000")

            a = parts[0]
            # Example: ge11nt112
            return [
                a[0:2],            # method (ge/po)
                a[2:4],            # version (11/20)
                a[4],              # cookie (c/n)
                a[5],              # referer (r/t)
                int(a[6:8]),        # header_cnt
                str(a[8:]),              # lang
                parts[1],          # b_hash
                parts[2],          # c_hash
            ]
        except:            
            return ['err', '00', 'n', 'n', 0, 'n', '000000000000', '000000000000']