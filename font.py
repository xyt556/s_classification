# ==================== 中文字体配置 ====================

# ==================== 中文字体配置 (优化版，适用于云端部署) ====================

def configure_chinese_fonts():
    """
    配置matplotlib中文字体，优先使用项目内置字体文件，确保云端部署时中文正常显示
    """
    import platform
    from matplotlib.font_manager import fontManager, FontProperties
    import os

    # ===== 策略1：优先加载项目内置字体（适用于部署环境） =====
    font_filename = 'SIMSUN.TTC'
    font_path = os.path.join('fonts', font_filename)

    if os.path.exists(font_path):
        try:
            # 动态注册字体到matplotlib
            fontManager.addfont(font_path)

            # 获取字体的实际名称
            prop = FontProperties(fname=font_path)
            font_name = prop.get_name()  # 通常是 'Source Han Sans SC'

            # 设置为matplotlib的默认字体
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

            return True, font_name
        except Exception as e:
            print(f"⚠️ 加载内置字体失败: {e}")
    else:
        print(f"⚠️ 未找到字体文件: {font_path}")

    # ===== 策略2：回退到系统字体（适用于本地开发） =====
    system = platform.system()
    chinese_fonts = []

    if system == 'Windows':
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun']
    elif system == 'Darwin':  # macOS
        chinese_fonts = ['PingFang SC', 'Heiti SC', 'STHeiti']
    else:  # Linux
        chinese_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback']

    from matplotlib.font_manager import FontManager
    fm = FontManager()
    available_fonts = {f.name for f in fm.ttflist}

    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break

    # 如果仍然找不到，尝试搜索包含中文关键词的字体
    if selected_font is None:
        for font in available_fonts:
            if any(keyword in font.lower() for keyword in ['chinese', 'cjk', 'han', 'hei', 'song']):
                selected_font = font
                break

    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font]
        plt.rcParams['axes.unicode_minus'] = False
        return True, selected_font
    else:
        print("❌ 未找到任何可用的中文字体")
        return False, None


# 执行字体配置
CHINESE_SUPPORT, SELECTED_FONT = configure_chinese_fonts()


# ==================== 文件上传限制 ====================